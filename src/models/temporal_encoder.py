"""Temporal encoder and cross-service attention for raw time-series processing.

Architecture Design Choices
===========================

**Why a temporal encoder at all?**
    The current CAAA pipeline collapses 5,040+ raw metric values (services x
    timesteps x metrics) into 44 scalar features. Tree baselines (CatBoost,
    XGBoost) can only process flat feature vectors. By adding a temporal encoder
    that processes the raw time-series tensor, CAAA gains access to information
    that tree models fundamentally cannot use — giving it a structural advantage.

**Why LITE-style 1D-CNN (not LSTM, Transformer, or GNN)?**
    1. *Parameter efficiency*: LITE (Ismail-Fawaz et al., DSAA 2023) achieves
       InceptionTime-level accuracy with only 9,814 parameters — critical for
       our ~1,470-sample training regime.
    2. *Small-data proven*: LITE outperforms InceptionTime on 71-sample datasets,
       demonstrating viability for our data scale.
    3. *LSTM alternative rejected*: BiLSTM would need ~30-50K params (Shallow
       BiLSTM) and is slower; recurrence is overkill for the pattern types we
       detect (spikes, trends, level shifts).
    4. *Transformer rejected*: Self-attention over 120 timesteps with 7 metrics
       would be ~100K+ params — too heavy. GTAD (2022) shows temporal
       convolution contributes similar signal with far fewer parameters.
    5. *GNN rejected*: ICSE 2025 ("Are GNNs Actually Effective?") showed DiagMLP
       matches/beats GNN methods — graph topology adds minimal discriminative
       value when features are well-engineered.

**Why depthwise separable convolutions?**
    Standard convolutions with 7 input channels and 8 output filters at
    kernel=15 require 7*8*15 = 840 params per branch. Depthwise separable
    splits this into depthwise (7*15 = 105) + pointwise (7*8 = 56) = 161
    params — 5x more efficient. Critical when the parameter budget is ~1,200.

**Why multi-scale kernels {3, 7, 15}?**
    Different fault types manifest at different temporal scales:
    - kernel=3: Local spikes, sudden errors (e.g., pod_failure, packet_loss)
    - kernel=7: Medium-range patterns (e.g., memory_leak ramp, network_delay)
    - kernel=15: Global trends (e.g., gradual CPU hog, seasonal load patterns)
    LITE paper validates that parallel multi-scale branches are more effective
    than a single deep stack of convolutions.

**Why cross-service attention (not GNN or mean pooling)?**
    1. *Attention over GNN*: ICSE 2025 negative result — GNN topology doesn't
       help when preprocessing is good. Self-attention learns inter-service
       relationships from data, no predefined graph needed.
    2. *Attention over mean pooling*: Not all services are equally informative.
       Attention lets the model focus on the 2-3 services showing anomalous
       behavior, ignoring healthy ones. GAL-MAD (2025) shows attention adds
       +10% recall over mean pooling.
    3. *Single head*: With d=16 and 12-20 services, multi-head attention would
       fragment the already-small representation. GTAD ablation shows the
       attention mechanism itself matters more than the number of heads.

References
----------
- LITE: Ismail-Fawaz et al. (2023). LITE: Light Inception with boosTing
  tEchniques for Time Series Classification. DSAA.
- GAL-MAD: arXiv:2504.00058 (2025). Graph Attention Layers for Microservice
  Anomaly Detection.
- Are GNNs Effective?: Yu et al. (2025). ICSE.
- GTAD: Entropy 2022. Graph and Temporal Neural Network for Anomaly Detection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """LITE-inspired multi-scale depthwise-separable 1D-CNN.

    Encodes a single service's metric time series into a fixed-size embedding.
    Shared across all services in a case (weight sharing).

    Architecture::

        Input (n_metrics, T)
            |
            +--- DWSConv(k=3)  ---> (n_filters, T)
            |
            +--- DWSConv(k=7)  ---> (n_filters, T)
            |
            +--- DWSConv(k=15) ---> (n_filters, T)
            |
            concat -> (3*n_filters, T)
            |
            LayerNorm + GELU
            |
            GlobalAvgPool -> (3*n_filters,)
            |
            Linear -> (d_service,)

    Args:
        n_metrics: Number of input metric channels (default 7).
        d_service: Output embedding dimensionality per service.
        n_filters: Number of filters per conv branch after pointwise projection.
        kernel_sizes: Tuple of kernel sizes for multi-scale branches.
    """

    def __init__(
        self,
        n_metrics: int = 7,
        d_service: int = 16,
        n_filters: int = 8,
        kernel_sizes: tuple = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                # Depthwise: each metric channel convolved independently
                nn.Conv1d(
                    n_metrics, n_metrics, kernel_size=k,
                    padding=k // 2, groups=n_metrics, bias=False,
                ),
                # Pointwise: mix across channels
                nn.Conv1d(n_metrics, n_filters, kernel_size=1, bias=False),
            )
            self.branches.append(branch)

        total_channels = n_filters * len(kernel_sizes)
        self.norm = nn.LayerNorm(total_channels)
        self.proj = nn.Linear(total_channels, d_service)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode service time series.

        Args:
            x: Tensor of shape ``(batch_services, n_metrics, T)``.

        Returns:
            Embedding of shape ``(batch_services, d_service)``.
        """
        branch_outs = [branch(x) for branch in self.branches]
        # Concat along channel dim -> (batch_svc, 3*n_filters, T)
        cat = torch.cat(branch_outs, dim=1)
        # Global average pool over time -> (batch_svc, 3*n_filters)
        pooled = cat.mean(dim=-1)
        # Normalize and project
        return self.proj(F.gelu(self.norm(pooled)))


class CrossServiceAttention(nn.Module):
    """Single-head self-attention over service embeddings.

    Aggregates per-service temporal embeddings into a single system-level
    embedding, allowing the model to focus on anomalous services.

    Uses a service mask to handle variable numbers of services across cases
    (padded service slots get -inf attention weights).

    **Why single-head**: With d=16 and 12-20 services, multi-head attention
    would fragment the representation. GTAD ablation confirms the attention
    mechanism itself matters more than the number of heads.

    **Why masked mean pool after attention** (not CLS token): CLS tokens
    require an extra learnable vector and work best with many tokens.
    With 12-20 services, masked mean pool is simpler and equally effective.

    Args:
        d_service: Dimensionality of service embeddings.
    """

    def __init__(self, d_service: int = 16) -> None:
        super().__init__()
        self.d = d_service
        self.qkv = nn.Linear(d_service, 3 * d_service, bias=False)
        self.out_proj = nn.Linear(d_service, d_service)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend across services and aggregate.

        Args:
            x: Service embeddings ``(batch, n_services, d_service)``.
            mask: Service mask ``(batch, n_services)``, 1.0 for real services.

        Returns:
            System-level embedding ``(batch, d_service)``.
        """
        B, S, D = x.shape
        qkv = self.qkv(x)  # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, S, D)

        # Scaled dot-product attention
        scale = math.sqrt(D)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale  # (B, S, S)

        # Mask: set padded positions to -inf
        mask_2d = mask.unsqueeze(1)  # (B, 1, S)
        attn = attn.masked_fill(mask_2d == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows (shouldn't happen but be safe)
        attn = attn.nan_to_num(0.0)

        attended = torch.bmm(attn, v)  # (B, S, D)
        out = self.out_proj(attended)  # (B, S, D)

        # Masked mean pool across services -> (B, D)
        mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
        pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled
