"""Context integration module for CAAA model.

This is the novel component that integrates context features
(event_active, event_expected_impact, time_seasonality,
recent_deployment, context_confidence) with temporal encoding
via FiLM conditioning and confidence gating.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContextIntegrationModule(nn.Module):
    """Integrates context features with temporal encoding.

    Uses FiLM (Feature-wise Linear Modulation) conditioning and confidence
    gating to selectively incorporate contextual information into the
    temporal representation.

    Components:
        1. Context encoder: Encodes 5 context features to hidden_dim.
        2. FiLM conditioning: Produces gamma/beta modulation parameters.
        3. Confidence gating: Gates context influence using multiple context signals.
        4. Residual connection: Adds input for gradient flow.

    Attributes:
        context_encoder: Encodes raw context features.
        film_gamma: Linear projection producing multiplicative modulation.
        film_beta: Linear projection producing additive modulation.
        confidence_gate: Gate driven by context_confidence, event_active,
            and recent_deployment.
        output_projection: Projects gated output back to temporal_dim.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        context_dim: int = 5,
        hidden_dim: int = 32,
        film_mode: str = "tadam",
    ) -> None:
        """Initializes the ContextIntegrationModule.

        Args:
            temporal_dim: Dimensionality of temporal encoding.
            context_dim: Number of context features (default 5).
            hidden_dim: Hidden dimensionality for context encoder.
            film_mode: FiLM conditioning mode. One of ``"multiplicative"``
                (original), ``"additive"`` (beta only), or ``"tadam"``
                (identity-initialised multiplicative delta).
        """
        super().__init__()
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.film_mode = film_mode
        self.last_delta = None

        # 1. Context encoder: context_dim -> hidden_dim -> hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. FiLM conditioning: gamma (scale) and beta (shift)
        self.film_gamma = nn.Linear(hidden_dim, temporal_dim)
        if self.film_mode == "tadam":
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
        self.film_beta = nn.Linear(hidden_dim, temporal_dim)

        # 3. Confidence gating (uses context_confidence, event_active, recent_deployment)
        self.confidence_gate = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )
        # Bias=-1.0 → initial sigmoid output ≈ 0.27, skeptical of context
        # by default.  The model must learn to trust context signal, preventing
        # over-reliance on noisy context features early in training.
        nn.init.constant_(self.confidence_gate[0].bias, -1.0)

        # 4. Output projection + residual
        self.output_projection = nn.Linear(temporal_dim, temporal_dim)

        logger.debug(
            "ContextIntegrationModule initialized: temporal_dim=%d, context_dim=%d, hidden_dim=%d",
            temporal_dim, context_dim, hidden_dim,
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass integrating temporal and context features.

        Args:
            temporal_features: Temporal encoding of shape (batch, temporal_dim).
            context_features: Context features of shape (batch, 5).

        Returns:
            Integrated features of shape (batch, temporal_dim).
        """
        residual = temporal_features

        # 1. Encode context features
        context_encoded = self.context_encoder(context_features)

        # 2. FiLM conditioning: modulate temporal features
        beta = self.film_beta(context_encoded)    # (batch, temporal_dim)
        if self.film_mode == "additive":
            modulated = temporal_features + beta
        elif self.film_mode == "tadam":
            delta = self.film_gamma(context_encoded)
            self.last_delta = delta
            gamma = 1.0 + delta
            modulated = gamma * temporal_features + beta
        else:  # "multiplicative" (original)
            gamma = self.film_gamma(context_encoded)
            modulated = gamma * temporal_features + beta

        # 3. Confidence gating using context_confidence, event_active, recent_deployment
        gate_input = torch.cat([
            context_features[:, -1:],  # context_confidence
            context_features[:, 0:1],  # event_active
            context_features[:, 3:4],  # recent_deployment
        ], dim=-1)  # (batch, 3)
        gate = self.confidence_gate(gate_input)  # (batch, 1)
        gated_output = gate * modulated + (1 - gate) * temporal_features

        # 4. Project and add residual
        projected = self.output_projection(gated_output)
        output = projected + residual

        return output
