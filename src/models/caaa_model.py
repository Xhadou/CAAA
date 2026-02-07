"""Main CAAA model combining temporal encoding with context integration."""

import logging

import torch
import torch.nn as nn

from src.models.context_module import ContextIntegrationModule
from src.models.temporal_encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class CAAAModel(nn.Module):
    """Context-Aware Anomaly Attribution model.

    Combines temporal encoding of feature vectors with context-aware
    integration to produce classification logits.

    Attributes:
        temporal_encoder: Encodes raw features into temporal representation.
        context_module: Integrates context features with temporal encoding.
        classifier: Classification head producing logits.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 64,
        context_dim: int = 5,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the CAAAModel.

        Args:
            input_dim: Dimensionality of input feature vectors.
            hidden_dim: Hidden dimensionality for temporal encoder.
            context_dim: Number of context features.
            n_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.context_module = ContextIntegrationModule(
            temporal_dim=hidden_dim,
            context_dim=context_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        logger.info(
            "CAAAModel initialized: input_dim=%d, hidden_dim=%d, n_classes=%d",
            input_dim, hidden_dim, n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            x: Input tensor of shape (batch, input_dim) containing
                the full 36-dim feature vector.

        Returns:
            Logits tensor of shape (batch, n_classes).
        """
        # Split context features (indices 12:17)
        context_features = x[:, 12:17]

        # Temporal encoding of full feature vector
        temporal_encoding = self.temporal_encoder(x)

        # Context integration
        integrated = self.context_module(temporal_encoding, context_features)

        # Classification
        logits = self.classifier(integrated)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class predictions (argmax of softmax).

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Predicted class indices of shape (batch,).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            return torch.argmax(probabilities, dim=-1)
