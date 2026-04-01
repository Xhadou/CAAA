"""Custom loss functions for CAAA training."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Reduces the loss contribution from easy examples, focusing
    training on hard misclassified examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ContextConsistencyLoss(nn.Module):
    """Custom loss combining classification, context consistency, and calibration.

    Components:
        1. Classification loss: Standard CrossEntropyLoss.
        2. Context consistency loss: Penalizes predictions contradicting
           available context signals.
        3. Confidence calibration loss: Penalizes high entropy when
           context confidence is high.

    Penalty scale note:
        When ``event_active ≈ 0.5`` and model probabilities are also
        ``≈ 0.5``, the per-sample penalty
        ``(penalty_when_event + penalty_when_no_event)`` approaches 1.0.
        With the default ``alpha=0.3``, this contributes up to 0.3 to
        the total loss — comparable in magnitude to the classification
        cross-entropy on well-separated data.  The consistency penalty
        therefore dominates the loss gradient early in training when the
        model is uncertain and the context signal is ambiguous.  As
        training progresses and predictions become more confident, the
        penalty shrinks because one of the two probability terms
        approaches zero.

    Attributes:
        alpha: Weight for the context consistency loss component.
        beta: Weight for the confidence calibration loss component.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        unknown_weight: float = 0.2,
        loss_variant: str = "gated",
    ) -> None:
        """Initializes the ContextConsistencyLoss.

        Args:
            alpha: Weight for context consistency loss.
            beta: Weight for confidence calibration loss.
            unknown_weight: Weight for the unknown-context penalty that
                penalizes EXPECTED_LOAD predictions when no event is active
                and context confidence is low.
            loss_variant: Controls unknown-context penalty behavior.
                ``"gated"`` (default) only penalizes high-confidence LOAD
                predictions without context.  ``"clamp"`` disables the
                penalty entirely.  ``"full"`` uses the original aggressive
                behavior.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.unknown_weight = unknown_weight
        self.loss_variant = loss_variant
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Args:
            logits: Model output logits of shape (batch, n_classes).
            labels: Ground truth labels of shape (batch,).
            context_features: Context features of shape (batch, 5).
                Indices: [event_active, event_expected_impact,
                time_seasonality, recent_deployment, context_confidence].

        Returns:
            Tuple of (total_loss, component_dict) where component_dict has
            keys 'cls_loss', 'consistency_loss', 'calibration_loss'.
        """
        # 1. Classification loss
        cls_loss = self.ce_loss(logits, labels)

        # 2. Context consistency loss
        probs = torch.softmax(logits, dim=-1)
        event_active = context_features[:, 0]  # 1.0 = load event happening
        context_confidence = context_features[:, 4]

        # When event_active == 1.0, penalize predicting FAULT (class 0)
        fault_prob = probs[:, 0]
        penalty_when_event = event_active * fault_prob

        # When event_active == 0.0, penalize predicting EXPECTED_LOAD (class 1)
        load_prob = probs[:, 1]
        penalty_when_no_event = (1.0 - event_active) * load_prob

        # Weight by clamped context_confidence so that low-confidence cases
        # (disguised faults, unscheduled loads) still produce learning signal.
        # Floor of 0.3 ensures disguised faults contribute 30% of the penalty
        # that high-confidence cases do.
        per_sample_penalty = penalty_when_event + penalty_when_no_event
        conf_weight = torch.clamp(context_confidence, min=0.3)
        consistency_loss = torch.mean(conf_weight * per_sample_penalty)

        # Unknown context penalty — variant-dependent
        unknown_context = (1.0 - event_active) * (1.0 - context_confidence)
        if self.loss_variant == "clamp":
            unknown_penalty = torch.tensor(0.0, device=logits.device)
        elif self.loss_variant == "gated":
            # Only penalize when model is very confident about LOAD without context
            gated_load_prob = torch.clamp(load_prob - 0.7, min=0.0)
            unknown_penalty = torch.mean(unknown_context * gated_load_prob)
        else:  # "full" — current aggressive behavior
            unknown_penalty = torch.mean(unknown_context * load_prob)

        # 3. Confidence calibration loss
        # When context_confidence is high, penalize high entropy (uncertainty)
        # Compute entropy of softmax distribution
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch,)
        calibration_loss = torch.mean(context_confidence * entropy)

        # Combined loss
        total_loss = (
            cls_loss
            + self.alpha * consistency_loss
            + self.beta * calibration_loss
            + self.unknown_weight * unknown_penalty
        )

        components = {
            "cls_loss": cls_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "calibration_loss": calibration_loss.item(),
            "unknown_penalty": unknown_penalty.item(),
        }

        return total_loss, components


class SupConContextLoss(nn.Module):
    """Supervised contrastive loss with context-aware temperature modulation.

    Same-class embeddings are pulled together while different-class embeddings
    are pushed apart.  Context confidence modulates the temperature: high
    confidence yields lower temperature (sharper contrast), while low
    confidence yields higher temperature (softer contrast).

    A small classification loss on the logits is added for end-to-end
    training of the classifier head.

    References:
        Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
        CARLA, Pattern Recognition 2024

    Attributes:
        base_temperature: Base temperature for contrastive softmax.
        context_weight: How much context confidence modulates temperature.
        cls_weight: Weight of the classification loss component.
    """

    def __init__(
        self,
        base_temperature: float = 0.07,
        context_weight: float = 0.3,
        cls_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_temperature = base_temperature
        self.context_weight = context_weight
        self.cls_weight = cls_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined contrastive + classification loss.

        Args:
            embeddings: Intermediate representations of shape
                ``(batch, hidden_dim)`` from ``model.get_embeddings()``.
            logits: Classification logits of shape ``(batch, n_classes)``.
            labels: Ground truth labels of shape ``(batch,)``.
            context_features: Context features of shape ``(batch, 5)``.

        Returns:
            Tuple of ``(total_loss, component_dict)`` where
            *component_dict* has keys ``contrastive_loss``, ``cls_loss``,
            and ``context_modulation``.
        """
        batch_size = embeddings.shape[0]

        # Per-sample temperature modulated by context_confidence
        # (the last context feature, index 4 within the 5-element context vector;
        # see src.features.feature_schema.CONTEXT_NAMES)
        ctx_conf = context_features[:, 4]
        temperature = self.base_temperature * (
            1.0 + self.context_weight * (1.0 - ctx_conf)
        )

        # Normalise embeddings to the unit hypersphere
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix (batch, batch)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Masks
        self_mask = 1.0 - torch.eye(batch_size, device=labels.device)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        positive_mask = label_mask * self_mask

        # Per-pair temperature: average of the two samples' temperatures
        pair_temp = (temperature.unsqueeze(0) + temperature.unsqueeze(1)) / 2.0

        # SupCon loss
        exp_sim = torch.exp(sim_matrix / pair_temp) * self_mask
        log_prob = sim_matrix / pair_temp - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )

        n_positives = positive_mask.sum(dim=1)
        # clamp(min=1) avoids huge gradients when n_positives=0: the
        # numerator is also 0 in that case, yielding a clean 0/1=0.
        contrastive_loss = -(positive_mask * log_prob).sum(dim=1) / (
            n_positives.clamp(min=1)
        )
        contrastive_loss = contrastive_loss.mean()

        # Classification loss for the classifier head
        cls_loss = self.ce_loss(logits, labels)

        total_loss = contrastive_loss + self.cls_weight * cls_loss

        components = {
            "contrastive_loss": contrastive_loss.item(),
            "cls_loss": cls_loss.item(),
            "context_modulation": temperature.mean().item(),
        }
        return total_loss, components
