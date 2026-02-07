"""Context integration module for CAAA model.

This is the novel component that integrates context features
(event_active, event_expected_impact, time_seasonality,
recent_deployment, context_confidence) with temporal encoding
via attention and confidence gating.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContextIntegrationModule(nn.Module):
    """Integrates context features with temporal encoding.

    Uses context-aware attention and confidence gating to selectively
    incorporate contextual information into the temporal representation.

    Components:
        1. Context encoder: Encodes 5 context features to hidden_dim.
        2. Context-aware attention: Computes attention over temporal encoding.
        3. Confidence gating: Gates context influence using context_confidence.
        4. Residual connection: Adds input for gradient flow.

    Attributes:
        context_encoder: Encodes raw context features.
        attention_layer: Produces attention weights from concatenated features.
        confidence_gate: Scalar gate driven by context_confidence.
        output_projection: Projects gated output back to temporal_dim.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        context_dim: int = 5,
        hidden_dim: int = 32,
    ) -> None:
        """Initializes the ContextIntegrationModule.

        Args:
            temporal_dim: Dimensionality of temporal encoding.
            context_dim: Number of context features (default 5).
            hidden_dim: Hidden dimensionality for context encoder.
        """
        super().__init__()
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # 1. Context encoder: context_dim -> hidden_dim -> hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. Context-aware attention
        self.attention_layer = nn.Sequential(
            nn.Linear(temporal_dim + hidden_dim, temporal_dim),
            nn.Sigmoid(),
        )

        # 3. Confidence gating
        self.confidence_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

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

        # 2. Context-aware attention
        combined = torch.cat([temporal_features, context_encoded], dim=-1)
        attention_weights = self.attention_layer(combined)
        attended = temporal_features * attention_weights

        # 3. Confidence gating using context_confidence (last context feature)
        context_confidence = context_features[:, -1:] # (batch, 1)
        gate = self.confidence_gate(context_confidence) # (batch, 1)
        gated_output = gate * attended + (1 - gate) * temporal_features

        # 4. Project and add residual
        projected = self.output_projection(gated_output)
        output = projected + residual

        return output
