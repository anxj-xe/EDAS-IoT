"""Baseline establishment and calibration module for EDAS-IoT."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ParameterStats(BaseModel):
    """Descriptive statistics for a single environmental metric."""
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float


class BaselineModel(BaseModel):
    """Learned environmental baseline state."""
    status: str = "BASELINE_NOT_READY" # "READY" or "BASELINE_NOT_READY"
    calibration_timestamp: Optional[str] = None
    dataset_name: Optional[str] = None
    sample_count: int = 0
    parameters: Dict[str, ParameterStats] = Field(default_factory=dict)
    epsilon: float = 1e-6

    def is_ready(self) -> bool:
        return self.status == "READY" and len(self.parameters) > 0


class BaselineCalibrator:
    """Calculates and manages environmental baselines."""

    def __init__(self, min_samples: int = 20, epsilon: float = 1e-6):
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.current_baseline = BaselineModel()

    def fit_from_dataframe(self, df: pd.DataFrame, dataset_name: str = "training_data.csv") -> BaselineModel:
        """Fit baseline statistics from a normal environmental dataset."""
        candidate_params = ["temperature_c", "humidity_pct", "pressure_hpa"]
        param_stats: Dict[str, ParameterStats] = {}

        total_valid_samples = 0
        for param in candidate_params:
            if param in df.columns:
                series = pd.to_numeric(df[param], errors="coerce").dropna()
                if len(series) >= self.min_samples:
                    mean_val = float(series.mean())
                    std_val = float(series.std(ddof=1))
                    # Prevent zero or near-zero standard deviation
                    if std_val < self.epsilon or np.isnan(std_val):
                        std_val = self.epsilon

                    param_stats[param] = ParameterStats(
                        count=int(len(series)),
                        mean=mean_val,
                        std=std_val,
                        min=float(series.min()),
                        max=float(series.max()),
                        median=float(series.median()),
                        q25=float(series.quantile(0.25)),
                        q75=float(series.quantile(0.75)),
                    )
                    total_valid_samples = max(total_valid_samples, len(series))

        if len(param_stats) == 0 or total_valid_samples < self.min_samples:
            self.current_baseline = BaselineModel(
                status="BASELINE_NOT_READY",
                calibration_timestamp=datetime.now(timezone.utc).isoformat(),
                dataset_name=dataset_name,
                sample_count=total_valid_samples,
                parameters={},
                epsilon=self.epsilon,
            )
            return self.current_baseline

        self.current_baseline = BaselineModel(
            status="READY",
            calibration_timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=dataset_name,
            sample_count=total_valid_samples,
            parameters=param_stats,
            epsilon=self.epsilon,
        )
        return self.current_baseline

    def save_baseline(self, filepath: Path) -> None:
        """Save baseline model to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.current_baseline.model_dump_json(indent=2))

    def load_baseline(self, filepath: Path) -> BaselineModel:
        """Load baseline model from a JSON file."""
        if not filepath.exists():
            self.current_baseline = BaselineModel(status="BASELINE_NOT_READY")
            return self.current_baseline

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.current_baseline = BaselineModel.model_validate(data)
            return self.current_baseline
