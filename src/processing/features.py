"""Feature engineering module for EDAS-IoT.

Generates consistent features for both model training (batch) and real-time inference (single reading).
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Extracts base and engineered environmental features with rolling statistics."""

    # Default core features
    BASE_FEATURES = ["temperature_c", "humidity_pct", "pressure_hpa"]

    def __init__(self, window_size: int = 5, use_rolling_features: bool = False):
        self.window_size = window_size
        self.use_rolling_features = use_rolling_features

        # Buffer for streaming features: Dict[device_id, Dict[param, deque]]
        self._buffers: Dict[str, Dict[str, deque]] = {}
        # Medians learned from baseline for explicit missing data imputation
        self.imputation_medians: Dict[str, float] = {
            "temperature_c": 22.0,
            "humidity_pct": 50.0,
            "pressure_hpa": 1013.25,
        }

    def set_imputation_medians(self, medians: Dict[str, float]) -> None:
        """Update imputation medians from the calibrated baseline."""
        self.imputation_medians.update(medians)

    def extract_batch_features(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a pandas DataFrame for model training or batch evaluation.

        Returns (X, feature_names).
        """
        target_cols = feature_cols or [col for col in self.BASE_FEATURES if col in df.columns]
        if not target_cols:
            raise ValueError(f"No valid feature columns found. Expected subset of {self.BASE_FEATURES}")

        df_copy = df.copy()

        # Handle missing data explicitly with median imputation and missing indicators
        feature_names = []
        for col in target_cols:
            median_val = self.imputation_medians.get(col, 0.0)
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            # If all are NaN, use median_val
            df_copy[col] = df_copy[col].fillna(median_val)
            feature_names.append(col)

        if self.use_rolling_features:
            for col in target_cols:
                diff_col = f"{col}_diff"
                df_copy[diff_col] = df_copy[col].diff().fillna(0.0)
                feature_names.append(diff_col)

                roll_mean_col = f"{col}_roll_mean"
                df_copy[roll_mean_col] = df_copy[col].rolling(window=self.window_size, min_periods=1).mean()
                feature_names.append(roll_mean_col)

        X = df_copy[feature_names].to_numpy(dtype=np.float32)
        return X, feature_names

    def extract_single_reading_features(
        self, device_id: str, reading_dict: Dict[str, Any], feature_cols: List[str]
    ) -> np.ndarray:
        """Extract feature vector for a single streaming reading."""
        if device_id not in self._buffers:
            self._buffers[device_id] = {col: deque(maxlen=self.window_size) for col in self.BASE_FEATURES}

        feature_values = []
        for col in feature_cols:
            # Check if this is a diff or roll feature
            if col.endswith("_diff"):
                base_col = col.replace("_diff", "")
                val = reading_dict.get(base_col)
                if val is None or np.isnan(val):
                    val = self.imputation_medians.get(base_col, 0.0)
                buf = self._buffers[device_id].get(base_col, deque())
                diff_val = (val - buf[-1]) if len(buf) > 0 else 0.0
                feature_values.append(diff_val)
            elif col.endswith("_roll_mean"):
                base_col = col.replace("_roll_mean", "")
                val = reading_dict.get(base_col)
                if val is None or np.isnan(val):
                    val = self.imputation_medians.get(base_col, 0.0)
                buf = self._buffers[device_id].get(base_col, deque())
                all_vals = list(buf) + [val]
                feature_values.append(float(np.mean(all_vals)))
            else:
                val = reading_dict.get(col)
                if val is None or np.isnan(val):
                    val = self.imputation_medians.get(col, 0.0)
                feature_values.append(float(val))

        # Update buffers with base values
        for base_col in self.BASE_FEATURES:
            val = reading_dict.get(base_col)
            if val is not None and not np.isnan(val):
                self._buffers[device_id][base_col].append(float(val))

        return np.array([feature_values], dtype=np.float32)

    def reset_state(self, device_id: Optional[str] = None) -> None:
        """Reset internal streaming state."""
        if device_id:
            self._buffers.pop(device_id, None)
        else:
            self._buffers.clear()
