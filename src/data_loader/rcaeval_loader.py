"""Load and parse RCAEval dataset format."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data_loader.data_types import AnomalyCase, ServiceMetrics

logger = logging.getLogger(__name__)


class RCAEvalLoader:
    """Load RCAEval datasets from disk.

    Expects the directory layout produced by
    :func:`src.data_loader.download_data.download_rcaeval_dataset`.
    """

    FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss", "socket"]

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_case_name(case_name: str) -> Dict:
        """Parse case directory name.

        Format: ``{system}_{service}_{fault}_{instance}``
        Example: ``online-boutique_frontend_cpu_1``
        """
        parts = case_name.split("_")
        return {
            "system": parts[0],
            "service": parts[1] if len(parts) > 1 else "unknown",
            "fault_type": parts[2] if len(parts) > 2 else "unknown",
            "instance": int(parts[3]) if len(parts) > 3 else 0,
        }

    @staticmethod
    def load_metrics(case_path: Path) -> pd.DataFrame:
        """Load the metrics CSV file from a case directory."""
        metrics_file = case_path / "metrics.csv"
        if not metrics_file.exists():
            for f in case_path.glob("*.csv"):
                if "metric" in f.name.lower():
                    metrics_file = f
                    break

        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if "timestamp" not in df.columns and df.columns[0].lower() in ["time", "ts"]:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            return df
        return pd.DataFrame()

    @staticmethod
    def load_logs(case_path: Path) -> Optional[pd.DataFrame]:
        """Load logs CSV if available."""
        logs_file = case_path / "logs.csv"
        return pd.read_csv(logs_file) if logs_file.exists() else None

    @staticmethod
    def load_traces(case_path: Path) -> Optional[pd.DataFrame]:
        """Load traces CSV if available."""
        traces_file = case_path / "traces.csv"
        return pd.read_csv(traces_file) if traces_file.exists() else None

    @staticmethod
    def load_ground_truth(case_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Load ground truth annotation if available."""
        gt_file = case_path / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file) as f:
                gt = json.load(f)
            return gt.get("root_cause_service"), gt.get("root_cause_metric")
        return None, None

    # ------------------------------------------------------------------
    # Wide-format parsing
    # ------------------------------------------------------------------

    # Known metric suffixes in RCAEval wide-format CSVs
    _KNOWN_METRICS = {
        "cpu_usage", "memory_usage", "request_rate", "error_rate",
        "latency", "network_in", "network_out",
    }

    @classmethod
    def parse_wide_format(cls, df: pd.DataFrame) -> List[ServiceMetrics]:
        """Parse wide-format metrics into per-service ServiceMetrics.

        Column names matching ``{service}_{metric}`` are split into
        separate DataFrames per unique service prefix.  Falls back to a
        single ServiceMetrics wrapping the entire DataFrame if no
        columns match the pattern.

        Args:
            df: Wide-format metrics DataFrame.

        Returns:
            List of ``ServiceMetrics``, one per detected service.
        """
        service_cols: Dict[str, List[str]] = defaultdict(list)
        unmatched: List[str] = []

        for col in df.columns:
            if col.lower() in ("timestamp", "time", "ts"):
                continue
            # Try to split as service_metric
            matched = False
            for metric in cls._KNOWN_METRICS:
                if col.endswith(f"_{metric}"):
                    svc_name = col[: -(len(metric) + 1)]
                    service_cols[svc_name].append((col, metric))
                    matched = True
                    break
            if not matched:
                unmatched.append(col)

        if not service_cols:
            # No service_metric pattern found — fall back to single entry
            return [ServiceMetrics(service_name="unknown", metrics=df)]

        # Build per-service DataFrames
        timestamp_col = None
        for candidate in ("timestamp", "time", "ts"):
            if candidate in df.columns:
                timestamp_col = candidate
                break

        results: List[ServiceMetrics] = []
        for svc_name, col_pairs in sorted(service_cols.items()):
            svc_df_data: Dict[str, object] = {}
            if timestamp_col:
                svc_df_data["timestamp"] = df[timestamp_col].values
            else:
                svc_df_data["timestamp"] = range(len(df))
            for orig_col, metric in col_pairs:
                svc_df_data[metric] = df[orig_col].values
            # Fill missing standard metrics with zeros
            for metric in cls._KNOWN_METRICS:
                if metric not in svc_df_data:
                    svc_df_data[metric] = 0.0
            svc_df = pd.DataFrame(svc_df_data)
            results.append(ServiceMetrics(service_name=svc_name, metrics=svc_df))

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset: str = "RE1",
        system: str = "online-boutique",
        fault_types: Optional[List[str]] = None,
    ) -> List[AnomalyCase]:
        """Load all failure cases from a dataset.

        Parses wide-format metrics into per-service ServiceMetrics when
        column names match the ``{service}_{metric}`` pattern.  Falls
        back to a single ServiceMetrics entry otherwise.

        Args:
            dataset: ``"RE1"``, ``"RE2"``, or ``"RE3"``.
            system: Microservice system name.
            fault_types: Filter by fault types. ``None`` returns all.

        Returns:
            List of ``AnomalyCase`` objects.
        """
        dataset_path = self.data_dir / dataset / system
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        cases: List[AnomalyCase] = []
        for case_dir in sorted(dataset_path.iterdir()):
            if not case_dir.is_dir():
                continue

            info = self.parse_case_name(case_dir.name)
            if fault_types and info["fault_type"] not in fault_types:
                continue

            metrics = self.load_metrics(case_dir)
            if metrics.empty:
                logger.warning(
                    "Empty metrics for case %s — skipping", case_dir.name,
                )
                continue

            # Parse wide-format into per-service metrics
            services = self.parse_wide_format(metrics)

            case = AnomalyCase(
                case_id=case_dir.name,
                system=info["system"],
                label="FAULT",
                services=services,
                fault_service=info["service"],
                fault_type=info["fault_type"],
            )
            cases.append(case)

        print(f"Loaded {len(cases)} failure cases from {dataset}/{system}")
        return cases


def load_rcaeval(dataset: str = "RE1", system: str = "online-boutique") -> List[AnomalyCase]:
    """Convenience loader for RCAEval data."""
    loader = RCAEvalLoader()
    return loader.load_dataset(dataset, system)


if __name__ == "__main__":
    cases = load_rcaeval("RE1", "online-boutique")
    print(f"Loaded {len(cases)} cases")
    if cases:
        print(f"First case: {cases[0].case_id}")
