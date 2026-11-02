"""Statistical anomaly detection module for EDAS-IoT.

Implements Z-score detection against baseline and rolling window statistical detection.
"""

from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
from src.common.config import get_config
from src.common.schemas import StatisticalResult
from src.detection.baseline import BaselineModel


class StatisticalAnomalyDetector:
    """Detects deviations using baseline Z-scores and short-term rolling statistics."""

    def __init__(
        self,
        baseline: Optional[BaselineModel] = None,
        z_score_threshold: float = 3.0,
        rolling_window_size: int = 10,
        rolling_z_threshold: float = 2.5,
        min_history_for_rolling: int = 5,
        epsilon: float = 1e-6,
    ):
        self.baseline = baseline
        self.z_score_threshold = z_score_threshold
        self.rolling_window_size = rolling_window_size
        self.rolling_z_threshold = rolling_z_threshold
        self.min_history_for_rolling = min_history_for_rolling
        self.epsilon = epsilon

        # Per-device rolling history: Dict[device_id, Dict[param, deque]]
        self._rolling_buffers: Dict[str, Dict[str, deque]] = {}

    def set_baseline(self, baseline: BaselineModel) -> None:
        """Update the active baseline model."""
        self.baseline = baseline

    def detect(self, device_id: str, readings: Dict[str, Optional[float]]) -> StatisticalResult:
        """Evaluate a reading dictionary for statistical anomalies."""
        z_scores: Dict[str, Optional[float]] = {}
        rolling_deviations: Dict[str, Optional[float]] = {}
        flagged_parameters: List[str] = []
        reasons: List[str] = []

        if device_id not in self._rolling_buffers:
            self._rolling_buffers[device_id] = {
                "temperature_c": deque(maxlen=self.rolling_window_size),
                "humidity_pct": deque(maxlen=self.rolling_window_size),
                "pressure_hpa": deque(maxlen=self.rolling_window_size),
            }

        # 1. Global Baseline Z-Score check
        baseline_ready = self.baseline is not None and self.baseline.is_ready()

        for param, val in readings.items():
            if val is None or np.isnan(val):
                z_scores[param] = None
                rolling_deviations[param] = None
                continue

            # A. Baseline Z-Score
            if baseline_ready and param in self.baseline.parameters:
                stats = self.baseline.parameters[param]
                std = max(stats.std, self.epsilon)
                z = (val - stats.mean) / std
                z_scores[param] = round(float(z), 3)

                if abs(z) >= self.z_score_threshold:
                    flagged_parameters.append(param)
                    direction = "above" if z > 0 else "below"
                    reasons.append(
                        f"{param} Z-score ({z:.2f}) is significantly {direction} baseline "
                        f"(mean={stats.mean:.1f}, std={stats.std:.2f}, limit=±{self.z_score_threshold:.1f})."
                    )
            else:
                z_scores[param] = None

            # B. Rolling Window Statistics
            buf = self._rolling_buffers[device_id].get(param)
            if buf is not None:
                if len(buf) >= self.min_history_for_rolling:
                    hist_vals = list(buf)
                    roll_mean = float(np.mean(hist_vals))
                    raw_roll_std = float(np.std(hist_vals, ddof=1)) if len(hist_vals) > 1 else 0.0

                    # Safeguard against tiny rolling variance: floor standard deviation
                    # using baseline std or sensible physical minimums
                    min_floor = 0.5
                    if baseline_ready and param in self.baseline.parameters:
                        min_floor = max(self.baseline.parameters[param].std * 0.4, 0.2)
                    roll_std = max(raw_roll_std, min_floor, self.epsilon)

                    roll_z = (val - roll_mean) / roll_std
                    rolling_deviations[param] = round(float(roll_z), 3)

                    # Only flag rolling anomaly if:
                    # 1. Rolling Z exceeds rolling threshold
                    # 2. Magnitude of shift is physically meaningful (>= min_floor)
                    # 3. The current observation is NOT simply returning to baseline normal
                    abs_diff = abs(val - roll_mean)
                    curr_z = z_scores.get(param)
                    is_within_normal_baseline = (curr_z is not None and abs(curr_z) < 2.0)

                    if (
                        abs(roll_z) >= self.rolling_z_threshold
                        and abs_diff >= min_floor
                        and not is_within_normal_baseline
                    ):
                        if param not in flagged_parameters:
                            flagged_parameters.append(param)
                        reasons.append(
                            f"Rapid shift in {param}: rolling deviation is {roll_z:.2f} relative to recent window "
                            f"(diff={abs_diff:.2f}, recent mean={roll_mean:.1f}, window={len(hist_vals)})."
                        )
                else:
                    rolling_deviations[param] = None

                # Append current observation to rolling buffer
                buf.append(val)

        is_anomaly = len(flagged_parameters) > 0

        return StatisticalResult(
            is_anomaly=is_anomaly,
            z_scores=z_scores,
            rolling_deviations=rolling_deviations,
            flagged_parameters=flagged_parameters,
            reasons=reasons,
        )

    def reset_state(self, device_id: Optional[str] = None) -> None:
        """Reset internal rolling buffers."""
        if device_id:
            self._rolling_buffers.pop(device_id, None)
        else:
            self._rolling_buffers.clear()
