"""Feature encoder module for CAAA model."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PiecewiseLinearEmbedding(nn.Module):
    """Transform scalar features into piecewise-linear bin encodings.

    Each scalar is encoded as a T-dimensional vector where each component
    is a soft indicator for one quantile bin. This gives MLPs axis-aligned
    threshold capability similar to tree splits.

    Reference: Gorishniy et al., NeurIPS 2022.
    """

    def __init__(self, n_features: int, n_bins: int = 8):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.register_buffer('bin_edges', torch.zeros(n_features, n_bins + 1))
        self._bins_set = False

    def set_bins(self, X_train):
        """Compute quantile-based bin edges from training data.

        Args:
            X_train: numpy array of shape (n_samples, n_features).
        """
        for j in range(self.n_features):
            col = X_train[:, j]
            edges = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
            # Ensure strictly increasing by adding small epsilon to duplicates
            for k in range(1, len(edges)):
                if edges[k] <= edges[k-1]:
                    edges[k] = edges[k-1] + 1e-8
            self.bin_edges[j] = torch.tensor(edges, dtype=torch.float32)
        self._bins_set = True

    @property
    def output_dim(self) -> int:
        return self.n_features * self.n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._bins_set:
            # Passthrough: return repeated features if bins not set (for testing)
            return x.repeat(1, self.n_bins)

        batch_size = x.shape[0]
        # Vectorized computation across all features
        # x: (batch, n_features) -> expand to (batch, n_features, 1)
        x_expanded = x.unsqueeze(2)  # (batch, n_features, 1)

        lo = self.bin_edges[:, :-1]  # (n_features, n_bins)
        hi = self.bin_edges[:, 1:]   # (n_features, n_bins)
        widths = (hi - lo).clamp(min=1e-8)  # (n_features, n_bins)

        # Broadcast: (batch, n_features, 1) - (n_features, n_bins) -> (batch, n_features, n_bins)
        enc = ((x_expanded - lo) / widths).clamp(0, 1)

        # Reshape to (batch, n_features * n_bins)
        return enc.reshape(batch_size, -1)


class FeatureEncoder(nn.Module):
    """Encodes the 44-dimensional feature vector into a dense hidden
    representation via a multi-layer perceptron.

    Note: despite operating on features derived from time series, this
    module does not perform sequential or temporal processing — temporal
    patterns are captured in the feature extraction stage.

    Attributes:
        layers: Sequential stack of linear transformation layers.
    """

    def __init__(
        self,
        input_dim: int = 44,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_ple: bool = True,
        n_bins: int = 8,
    ) -> None:
        """Initializes the FeatureEncoder.

        Args:
            input_dim: Dimensionality of input feature vectors.
            hidden_dim: Dimensionality of hidden layers.
            num_layers: Number of Linear + LayerNorm + GELU + Dropout blocks.
            dropout: Dropout probability.
            use_ple: Whether to apply PiecewiseLinearEmbedding before MLP layers.
            n_bins: Number of quantile bins for PLE (used only when use_ple=True).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_ple = use_ple

        if use_ple:
            self.ple = PiecewiseLinearEmbedding(input_dim, n_bins)
            first_layer_dim = input_dim * n_bins
        else:
            first_layer_dim = input_dim

        layers: list[nn.Module] = []
        # First layer: first_layer_dim -> hidden_dim
        layers.extend([
            nn.Linear(first_layer_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)
        logger.debug(
            "FeatureEncoder initialized: input_dim=%d, hidden_dim=%d, num_layers=%d, use_ple=%s",
            input_dim, hidden_dim, num_layers, use_ple,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim).
        """
        if self.use_ple:
            x = self.ple(x)
        return self.layers(x)

    def set_bins(self, X_train) -> None:
        """Set PLE bin edges from training data.

        Args:
            X_train: numpy array of shape (n_samples, n_features).
        """
        if self.use_ple:
            self.ple.set_bins(X_train)
