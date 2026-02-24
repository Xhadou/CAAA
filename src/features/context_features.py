"""Extract context features from external event information.

This is the NOVEL component — integrating business context
(event calendars, deployments, time-of-day) into anomaly classification.

Note:
    The ``ContextFeatureExtractor`` and ``EventCalendar`` classes were removed
    because they were unused in the pipeline.  Context features are computed
    inline in :mod:`src.features.extractors` (``_context_features``).
    The ``ContextFeatures`` dataclass is retained as a typed container.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ContextFeatures:
    """Container for context-related features."""

    event_active: float
    event_expected_impact: float
    time_seasonality: float
    recent_deployment: float
    context_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "event_active": self.event_active,
            "event_expected_impact": self.event_expected_impact,
            "time_seasonality": self.time_seasonality,
            "recent_deployment": self.recent_deployment,
            "context_confidence": self.context_confidence,
        }
