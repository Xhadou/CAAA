"""Load and parse RCAEval dataset format."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader.data_types import AnomalyCase, ServiceMetrics

logger = logging.getLogger(__name__)


class RCAEvalLoader:
    """Load RCAEval datasets from disk.

    Expects the directory layout produced by
    :func:`src.data_loader.download_data.download_rcaeval_dataset`.
    """

    FAULT_TYPES = [
        "cpu", "mem", "disk", "delay", "loss", "socket",
        # RE3 uses generic fault IDs
        "f1", "f2", "f3", "f4", "f5",
    ]

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def parse_case_name(cls, case_name: str) -> Dict:
        """Parse case directory name.

        Supports two formats:
        - Long:  ``{system}_{service}_{fault}_{instance}``
          e.g.   ``online-boutique_frontend_cpu_1``
        - Short: ``{service}_{fault}``
          e.g.   ``adservice_cpu``  (used in nested RCAEval layout)
        """
        parts = case_name.split("_")

        # Short format: last part is fault type if it matches a known fault
        if len(parts) == 2 and parts[-1] in cls.FAULT_TYPES:
            return {
                "system": "unknown",
                "service": parts[0],
                "fault_type": parts[1],
                "instance": 0,
            }

        # Try to detect short format with multi-word service name
        # e.g. "productcatalogservice_cpu"
        for i, part in enumerate(parts):
            if part in cls.FAULT_TYPES:
                return {
                    "system": "unknown",
                    "service": "_".join(parts[:i]),
                    "fault_type": part,
                    "instance": int(parts[i + 1]) if i + 1 < len(parts) else 0,
                }

        # Long format fallback
        return {
            "system": parts[0],
            "service": parts[1] if len(parts) > 1 else "unknown",
            "fault_type": parts[2] if len(parts) > 2 else "unknown",
            "instance": int(parts[3]) if len(parts) > 3 else 0,
        }

    @staticmethod
    def load_metrics(case_path: Path) -> pd.DataFrame:
        """Load the metrics CSV file from a case directory.

        Prefers ``simple_metrics.csv`` / ``simple_data.csv`` (clean
        ``{service}_{metric}`` columns) over the raw ``metrics.csv`` /
        ``data.csv`` which may contain hundreds of hyphenated Prometheus-
        style columns that the wide-format parser cannot split.
        """
        # Priority order: simple formats first, then raw formats
        candidates = [
            case_path / "simple_metrics.csv",
            case_path / "simple_data.csv",
            case_path / "metrics.csv",
            case_path / "data.csv",
        ]
        metrics_file = None
        for candidate in candidates:
            if candidate.exists():
                metrics_file = candidate
                break

        if metrics_file is None:
            # Last resort: any CSV with "metric" or "data" in the name
            for f in case_path.glob("*.csv"):
                if "metric" in f.name.lower() or "data" in f.name.lower():
                    metrics_file = f
                    break

        if metrics_file is not None and metrics_file.exists():
            df = pd.read_csv(metrics_file)
            # Drop duplicate columns (e.g. second "time" column in RCAEval CSVs)
            df = df.loc[:, ~df.columns.duplicated()]
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

    # Known metric suffixes in RCAEval wide-format CSVs.
    # Maps raw suffix → standardised column name used by ServiceMetrics.
    _METRIC_ALIASES: Dict[str, str] = {
        # Standard names (synthetic generator format)
        "cpu_usage": "cpu_usage",
        "memory_usage": "memory_usage",
        "request_rate": "request_rate",
        "error_rate": "error_rate",
        "latency": "latency",
        "network_in": "network_in",
        "network_out": "network_out",
        # Short names used in RCAEval CSVs
        "cpu": "cpu_usage",
        "mem": "memory_usage",
        "load": "request_rate",
        "error": "error_rate",
        # Additional RCAEval simple-format suffixes (RE2/RE3)
        "workload": "request_rate",
        "latency-50": "latency",
        "latency-90": "latency",  # prefer p90; overwrites p50 if both present
        "diskio": "network_in",   # map to network_in as proxy for I/O
        "socket": "network_out",  # map to network_out as proxy for connections
    }
    # Sort longest-first so "latency-90" is tried before "latency"
    _KNOWN_METRICS_ORDERED = sorted(_METRIC_ALIASES.keys(), key=len, reverse=True)
    _KNOWN_METRICS = set(_METRIC_ALIASES.keys())

    # The canonical metric columns expected by downstream code
    _STANDARD_METRICS = {
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
            for metric in cls._KNOWN_METRICS_ORDERED:
                if col.endswith(f"_{metric}"):
                    svc_name = col[: -(len(metric) + 1)]
                    std_metric = cls._METRIC_ALIASES[metric]
                    service_cols[svc_name].append((col, std_metric))
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
            for orig_col, std_metric in col_pairs:
                svc_df_data[std_metric] = df[orig_col].values
            # Fill missing standard metrics with zeros
            for metric in cls._STANDARD_METRICS:
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

        # Discover case directories.  The layout may be either:
        #   <dataset_path>/<case_name>/metrics.csv          (flat)
        #   <dataset_path>/<prefix>/<svc_fault>/<instance>/ (nested, e.g. RE1-OB/adservice_cpu/1/)
        # We detect the nested layout by checking whether the first-level
        # children themselves contain case-level sub-directories.
        case_dirs: List[Path] = []
        for child in sorted(dataset_path.iterdir()):
            if not child.is_dir():
                continue
            # Check if this child is a nested prefix dir (contains dirs that
            # themselves contain data files or numbered instance dirs).
            # Avoid loading CSVs just for layout detection — check for files instead.
            sub_dirs = [d for d in sorted(child.iterdir()) if d.is_dir()]
            has_own_csv = any(child.glob("*.csv"))
            if sub_dirs and not has_own_csv:
                # Nested layout: child is a prefix like RE1-OB
                for svc_fault_dir in sub_dirs:
                    # Each svc_fault_dir may contain numbered instance dirs
                    instance_dirs = [d for d in sorted(svc_fault_dir.iterdir()) if d.is_dir()]
                    if instance_dirs:
                        for inst_dir in instance_dirs:
                            case_dirs.append(inst_dir)
                    else:
                        case_dirs.append(svc_fault_dir)
            else:
                case_dirs.append(child)

        for case_dir in case_dirs:
            # Derive case name: for nested layouts use parent dir name
            # (e.g. "adservice_cpu") to get service/fault info.
            if case_dir.parent.parent != dataset_path:
                # Nested: case_dir = .../RE1-OB/adservice_cpu/1/
                case_name = case_dir.parent.name
            else:
                case_name = case_dir.name

            info = self.parse_case_name(case_name)
            # For nested layouts the system comes from the path, not the dir name
            if info["system"] == "unknown":
                info["system"] = system
            if fault_types and info["fault_type"] not in fault_types:
                continue

            metrics = self.load_metrics(case_dir)
            if metrics.empty:
                logger.warning(
                    "Empty metrics for case %s — skipping", case_dir.name,
                )
                continue

            # Load fault injection time if available
            inject_time = None
            inject_file = case_dir / "inject_time.txt"
            if inject_file.exists():
                try:
                    inject_time = int(inject_file.read_text().strip())
                except (ValueError, OSError):
                    pass

            # Parse wide-format into per-service metrics
            services = self.parse_wide_format(metrics)

            case_id = f"{case_name}_{case_dir.name}" if case_dir.parent.parent != dataset_path else case_dir.name
            case = AnomalyCase(
                case_id=case_id,
                system=info["system"],
                label="FAULT",
                services=services,
                fault_service=info["service"],
                fault_type=info["fault_type"],
            )
            cases.append(case)

        print(f"Loaded {len(cases)} failure cases from {dataset}/{system}")
        return cases

    def load_dataset_split(
        self,
        dataset: str = "RE1",
        system: str = "online-boutique",
        fault_types: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
        """Load RCAEval cases split at injection time into FAULT and NORMAL halves.

        For each case directory, reads the metrics CSV and ``inject_time.txt``,
        then splits the metrics DataFrame at the inject_time timestamp.  The
        post-injection window becomes a FAULT case; the pre-injection window
        becomes an EXPECTED_LOAD (normal) case.  Both halves originate from the
        same real recording, avoiding distribution mismatch between FAULT and
        NORMAL data.

        Context is randomized to prevent label leakage: 70% of NORMAL cases
        get context metadata, 30% get empty context.  30% of FAULT cases get
        fake context, 70% get empty context.  This prevents ``event_active``
        from being a perfect label proxy.

        Cases without ``inject_time.txt`` are included as FAULT-only (no NORMAL
        pair).  Splits where either half has fewer than 10 rows are skipped.

        Args:
            dataset: ``"RE1"``, ``"RE2"``, or ``"RE3"``.
            system: Microservice system name.
            fault_types: Filter by fault types. ``None`` returns all.
            seed: Random seed for reproducible context randomization.

        Returns:
            Tuple of ``(fault_cases, normal_cases)`` as
            ``Tuple[List[AnomalyCase], List[AnomalyCase]]``.
        """
        dataset_path = self.data_dir / dataset / system
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        rng = np.random.default_rng(seed)
        fault_cases: List[AnomalyCase] = []
        normal_cases: List[AnomalyCase] = []

        # ---- directory discovery (mirrors load_dataset) ----
        case_dirs: List[Path] = []
        for child in sorted(dataset_path.iterdir()):
            if not child.is_dir():
                continue
            sub_dirs = [d for d in sorted(child.iterdir()) if d.is_dir()]
            has_own_csv = any(child.glob("*.csv"))
            if sub_dirs and not has_own_csv:
                for svc_fault_dir in sub_dirs:
                    instance_dirs = [d for d in sorted(svc_fault_dir.iterdir()) if d.is_dir()]
                    if instance_dirs:
                        for inst_dir in instance_dirs:
                            case_dirs.append(inst_dir)
                    else:
                        case_dirs.append(svc_fault_dir)
            else:
                case_dirs.append(child)

        for case_dir in case_dirs:
            if case_dir.parent.parent != dataset_path:
                case_name = case_dir.parent.name
            else:
                case_name = case_dir.name

            info = self.parse_case_name(case_name)
            if info["system"] == "unknown":
                info["system"] = system
            if fault_types and info["fault_type"] not in fault_types:
                continue

            metrics = self.load_metrics(case_dir)
            if metrics.empty:
                logger.warning("Empty metrics for case %s — skipping", case_dir.name)
                continue

            case_id = (
                f"{case_name}_{case_dir.name}"
                if case_dir.parent.parent != dataset_path
                else case_dir.name
            )

            # ---- read inject_time ----
            inject_time = None
            inject_file = case_dir / "inject_time.txt"
            if inject_file.exists():
                try:
                    inject_time = int(inject_file.read_text().strip())
                except (ValueError, OSError):
                    pass

            if inject_time is None:
                # No split possible — emit FAULT case for the full window
                services = self.parse_wide_format(metrics)
                fault_cases.append(
                    AnomalyCase(
                        case_id=case_id,
                        system=info["system"],
                        label="FAULT",
                        services=services,
                        context={},
                        fault_service=info["service"],
                        fault_type=info["fault_type"],
                    )
                )
                continue

            # ---- split on inject_time ----
            # Determine the timestamp column
            ts_col = None
            for candidate in ("timestamp", "time", "ts"):
                if candidate in metrics.columns:
                    ts_col = candidate
                    break

            if ts_col is not None:
                pre_df = metrics[metrics[ts_col] < inject_time].reset_index(drop=True)
                post_df = metrics[metrics[ts_col] >= inject_time].reset_index(drop=True)
            else:
                # No timestamp column — split by row index at the midpoint
                # (inject_time is treated as a row index)
                pre_df = metrics.iloc[:inject_time].reset_index(drop=True)
                post_df = metrics.iloc[inject_time:].reset_index(drop=True)

            # Skip if either half is too short
            if len(pre_df) < 10 or len(post_df) < 10:
                logger.debug(
                    "Skipping split for %s: pre=%d rows, post=%d rows",
                    case_id, len(pre_df), len(post_df),
                )
                continue

            # FAULT case: post-injection window
            # 30% get fake context to prevent event_active from being
            # a perfect label proxy (mirrors synthetic data's 30% rate).
            fault_services = self.parse_wide_format(post_df)
            reference_services = self.parse_wide_format(pre_df)
            if rng.random() < 0.30:
                fault_context = {
                    "event_type": "maintenance_window",
                    "context_confidence": 0.6,
                }
            else:
                fault_context = {}
            fault_cases.append(
                AnomalyCase(
                    case_id=f"{case_id}_fault",
                    system=info["system"],
                    label="FAULT",
                    services=fault_services,
                    context=fault_context,
                    fault_service=info["service"],
                    fault_type=info["fault_type"],
                    reference_services=reference_services,
                )
            )

            # NORMAL case: pre-injection window
            # 70% get context, 30% get empty context to prevent
            # context presence from being a perfect label proxy.
            # Split pre-injection: first 25% as reference, last 75% as case
            ref_split = int(len(pre_df) * 0.25)
            if ref_split < 10:
                # Too few rows for reference — use full pre_df as both
                normal_services = self.parse_wide_format(pre_df)
                normal_reference = normal_services
            else:
                ref_df = pre_df.iloc[:ref_split].reset_index(drop=True)
                case_df = pre_df.iloc[ref_split:].reset_index(drop=True)
                normal_services = self.parse_wide_format(case_df)
                normal_reference = self.parse_wide_format(ref_df)
            if rng.random() > 0.30:
                normal_context = {
                    "event_type": "normal_operation",
                    "context_confidence": 0.8,
                }
            else:
                normal_context = {}
            normal_cases.append(
                AnomalyCase(
                    case_id=f"{case_id}_normal",
                    system=info["system"],
                    label="EXPECTED_LOAD",
                    services=normal_services,
                    context=normal_context,
                    reference_services=normal_reference,
                )
            )

        logger.info(
            "load_dataset_split: %d fault cases, %d normal cases from %s/%s",
            len(fault_cases), len(normal_cases), dataset, system,
        )
        return fault_cases, normal_cases


def load_rcaeval(dataset: str = "RE1", system: str = "online-boutique") -> List[AnomalyCase]:
    """Convenience loader for RCAEval data."""
    loader = RCAEvalLoader()
    return loader.load_dataset(dataset, system)


if __name__ == "__main__":
    cases = load_rcaeval("RE1", "online-boutique")
    print(f"Loaded {len(cases)} cases")
    if cases:
        print(f"First case: {cases[0].case_id}")
