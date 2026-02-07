"""Data types for the CAAA anomaly attribution pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ServiceMetrics:
    """Holds time-series metrics for a single microservice.

    Attributes:
        service_name: Name of the microservice.
        metrics: DataFrame with columns: timestamp, cpu_usage, memory_usage,
            request_rate, error_rate, latency, network_in, network_out.
    """

    service_name: str
    metrics: pd.DataFrame


@dataclass
class AnomalyCase:
    """Represents a single anomaly case for classification.

    Attributes:
        case_id: Unique identifier for this case.
        system: Name of the system under observation.
        label: Classification label, either "FAULT" or "EXPECTED_LOAD".
        services: List of ServiceMetrics for all observed services.
        context: Optional dictionary of contextual information.
        fault_service: Optional name of the faulty service (for FAULT cases).
        fault_type: Optional type of fault injected (for FAULT cases).
    """

    case_id: str
    system: str
    label: str
    services: List[ServiceMetrics]
    context: Optional[Dict] = field(default_factory=dict)
    fault_service: Optional[str] = None
    fault_type: Optional[str] = None
