"""Main CAAA model combining feature encoding with context integration.

When ``use_temporal=True``, the model additionally processes raw time-series
tensors through a lightweight temporal encoder and cross-service attention
module, fusing the resulting system-level embedding with the flat feature
vector before the main encoder.  This gives the model access to temporal
dynamics and inter-service interactions that tree baselines cannot exploit.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.features.feature_schema import CONTEXT_START, CONTEXT_END, N_FEATURES
from src.models.context_module import ContextIntegrationModule
from src.models.feature_encoder import FeatureEncoder

logger = logging.getLogger(__name__)


class CAAAModel(nn.Module):
    """Context-Aware Anomaly Attribution model.

    Combines MLP-based feature encoding with context-aware integration
    to produce classification logits.  Optionally includes a temporal
    branch that processes raw per-service metric time series.

    Attributes:
        feature_encoder: MLP that projects features into a dense
            hidden representation.
        context_module: Integrates context features with the encoded
            representation via attention and confidence gating.
        classifier: Classification head producing logits.
        use_temporal: Whether the temporal branch is active.
    """

    def __init__(
        self,
        input_dim: int = 44,
        hidden_dim: int = 64,
        context_dim: int = 5,
        n_classes: int = 2,
        dropout: float = 0.1,
        film_mode: str = "tadam",
        use_temporal: bool = False,
        d_service: int = 16,
    ) -> None:
        """Initializes the CAAAModel.

        Args:
            input_dim: Dimensionality of input feature vectors (44 for
                the standard flat feature pipeline).
            hidden_dim: Hidden dimensionality for feature encoder.
            context_dim: Number of context features.
            n_classes: Number of output classes.
            dropout: Dropout probability.
            film_mode: FiLM conditioning mode passed to
                :class:`ContextIntegrationModule`. One of
                ``"multiplicative"``, ``"additive"``, or ``"tadam"``.
            use_temporal: Whether to enable the temporal encoder branch
                that processes raw ``(services, timesteps, metrics)``
                tensors. When False (default), the model behaves
                identically to the original CAAA architecture.
            d_service: Dimensionality of per-service temporal embeddings.
                Only used when ``use_temporal=True``.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        self.d_service = d_service

        # Temporal branch (optional)
        if use_temporal:
            from src.models.temporal_encoder import (
                TemporalEncoder,
                CrossServiceAttention,
            )
            self.temporal_encoder = TemporalEncoder(d_service=d_service)
            self.cross_service_attn = CrossServiceAttention(d_service=d_service)
            fused_dim = input_dim + d_service
            ple_dim = input_dim  # PLE only on original features, not temporal
        else:
            fused_dim = input_dim
            ple_dim = 0  # 0 = apply PLE to all dims (existing behavior)

        self.feature_encoder = FeatureEncoder(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ple_dim=ple_dim,
        )
        self.context_module = ContextIntegrationModule(
            temporal_dim=hidden_dim,
            context_dim=context_dim,
            film_mode=film_mode,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        mode_str = "temporal" if use_temporal else "flat"
        logger.info(
            "CAAAModel initialized: input_dim=%d, hidden_dim=%d, n_classes=%d, mode=%s",
            input_dim, hidden_dim, n_classes, mode_str,
        )

    def _encode_temporal(
        self,
        raw_tensor: torch.Tensor,
        service_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw time-series tensor into a system-level embedding.

        Args:
            raw_tensor: Shape ``(batch, max_services, T, n_metrics)``.
            service_mask: Shape ``(batch, max_services)``, 1.0 for real.

        Returns:
            System embedding of shape ``(batch, d_service)``.
        """
        B, S, T, M = raw_tensor.shape
        # Reshape for shared temporal encoder: (B*S, n_metrics, T)
        flat = raw_tensor.reshape(B * S, T, M).permute(0, 2, 1)
        svc_emb = self.temporal_encoder(flat)  # (B*S, d_service)
        svc_emb = svc_emb.reshape(B, S, -1)  # (B, S, d_service)
        # Zero out padded services before attention
        svc_emb = svc_emb * service_mask.unsqueeze(-1)
        return self.cross_service_attn(svc_emb, service_mask)  # (B, d_service)

    def forward(
        self,
        x: torch.Tensor,
        raw_tensor: Optional[torch.Tensor] = None,
        service_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the full model.

        The full 44-dim feature vector (including context dims 12-16) is
        passed through the FeatureEncoder so that the encoder can learn
        joint representations that capture interactions between context
        and metric features.  Context features are *also* sliced out
        separately and fed into the ContextIntegrationModule for
        explicit attention and confidence gating.

        When ``use_temporal=True`` and ``raw_tensor`` is provided, the raw
        time-series is encoded into a d_service-dim system embedding and
        concatenated with the flat features before the feature encoder.

        Args:
            x: Input tensor of shape ``(batch, input_dim)`` containing
                the flat feature vector (44-dim for standard pipeline).
            raw_tensor: Optional raw time-series tensor of shape
                ``(batch, max_services, T, n_metrics)``. Only used when
                ``use_temporal=True``.
            service_mask: Optional service mask of shape
                ``(batch, max_services)``. Required when raw_tensor is
                provided.

        Returns:
            Logits tensor of shape ``(batch, n_classes)``.
        """
        # Fuse temporal embedding with flat features if temporal branch active
        if self.use_temporal and raw_tensor is not None:
            sys_emb = self._encode_temporal(raw_tensor, service_mask)
            x = torch.cat([x, sys_emb], dim=-1)

        # Context features are always in the first input_dim dims
        context_features = x[:, CONTEXT_START:CONTEXT_END]

        # Feature encoding of (possibly fused) feature vector
        encoded_features = self.feature_encoder(x)

        # Context integration via FiLM conditioning
        integrated = self.context_module(encoded_features, context_features)

        # Classification
        return self.classifier(integrated)

    def get_embeddings(
        self,
        x: torch.Tensor,
        raw_tensor: Optional[torch.Tensor] = None,
        service_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return intermediate embeddings before the classifier head.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            raw_tensor: Optional raw tensor (see :meth:`forward`).
            service_mask: Optional service mask (see :meth:`forward`).

        Returns:
            Embedding tensor of shape ``(batch, hidden_dim)``.
        """
        if self.use_temporal and raw_tensor is not None:
            sys_emb = self._encode_temporal(raw_tensor, service_mask)
            x = torch.cat([x, sys_emb], dim=-1)
        context_features = x[:, CONTEXT_START:CONTEXT_END]
        encoded_features = self.feature_encoder(x)
        return self.context_module(encoded_features, context_features)

    def set_bins(self, X_train) -> None:
        """Set PLE bin edges from training data.

        Args:
            X_train: numpy array of shape ``(n_samples, n_features)``.
                When temporal is active, only the first ``input_dim``
                columns are used (temporal embedding dims are skipped).
        """
        self.feature_encoder.set_bins(X_train)

    def predict(
        self,
        x: torch.Tensor,
        raw_tensor: Optional[torch.Tensor] = None,
        service_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns class predictions (argmax of softmax).

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            raw_tensor: Optional raw tensor (see :meth:`forward`).
            service_mask: Optional service mask (see :meth:`forward`).

        Returns:
            Predicted class indices of shape ``(batch,)``.
        """
        with torch.no_grad():
            logits = self.forward(x, raw_tensor, service_mask)
            probabilities = torch.softmax(logits, dim=-1)
            return torch.argmax(probabilities, dim=-1)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.6,
        raw_tensor: Optional[torch.Tensor] = None,
        service_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions with UNKNOWN class for low-confidence ones.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            confidence_threshold: Minimum softmax probability to accept.
            raw_tensor: Optional raw tensor (see :meth:`forward`).
            service_mask: Optional service mask (see :meth:`forward`).

        Returns:
            Tuple of (predictions, confidences).
        """
        with torch.no_grad():
            logits = self.forward(x, raw_tensor, service_mask)
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
            predictions[confidences < confidence_threshold] = 2
        return predictions, confidences
