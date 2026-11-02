"""Isolation Forest machine learning anomaly detection module for EDAS-IoT."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.common.schemas import MLModelResult
from src.processing.features import FeatureEngineer


class IsolationForestDetector:
    """Wrapper around Scikit-Learn IsolationForest with model persistence and structured output."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        decision_threshold: float = -0.05,
        random_state: int = 42,
        features: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        self.features = features or ["temperature_c", "humidity_pct", "pressure_hpa"]

        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False
        self.version: str = "1.0.0"
        self.training_metadata: Dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer()

    def fit(self, df: pd.DataFrame, version: str = "1.0.0") -> Dict[str, Any]:
        """Fit Isolation Forest model on normal environmental readings."""
        self.version = version

        # Determine features that exist in df
        actual_features = [f for f in self.features if f in df.columns]
        if not actual_features:
            raise ValueError(f"None of the specified features {self.features} found in DataFrame columns: {df.columns.tolist()}")

        self.features = actual_features

        # Compute and record medians for imputation
        medians = {}
        for feat in self.features:
            series = pd.to_numeric(df[feat], errors="coerce")
            medians[feat] = float(series.median()) if not series.dropna().empty else 0.0
        self.feature_engineer.set_imputation_medians(medians)

        # Extract training array
        X, _ = self.feature_engineer.extract_batch_features(df, self.features)

        # Initialize and fit scikit-learn IsolationForest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.is_fitted = True

        # Decision scores on training data
        train_scores = self.model.decision_function(X)
        score_min = float(np.min(train_scores))
        score_max = float(np.max(train_scores))
        score_mean = float(np.mean(train_scores))

        self.training_metadata = {
            "version": self.version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(X)),
            "features": self.features,
            "imputation_medians": medians,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "decision_threshold": self.decision_threshold,
            "random_state": self.random_state,
            "score_stats": {
                "min": score_min,
                "max": score_max,
                "mean": score_mean,
            },
        }

        return self.training_metadata

    def predict_single(self, device_id: str, reading_dict: Dict[str, Any]) -> MLModelResult:
        """Run inference on a single reading dictionary."""
        if not self.is_fitted or self.model is None:
            # Fallback when model is not yet trained
            return MLModelResult(
                model_name="isolation_forest",
                is_anomaly=False,
                decision_score=0.0,
                anomaly_score=0.5,
            )

        X = self.feature_engineer.extract_single_reading_features(device_id, reading_dict, self.features)
        decision_score = float(self.model.decision_function(X)[0])

        # Scikit-learn score: lower means more abnormal. Compare against calibrated threshold
        is_anomaly = bool(decision_score < self.decision_threshold)

        # Map decision score to roughly 0 (very normal) to 1 (highly anomalous)
        # Normal decision scores typically range from -0.3 (outlier) to +0.3 (inlier)
        normalized_anomaly_score = float(np.clip(0.5 - (decision_score * 1.5), 0.0, 1.0))

        return MLModelResult(
            model_name="isolation_forest",
            is_anomaly=is_anomaly,
            decision_score=round(decision_score, 4),
            anomaly_score=round(normalized_anomaly_score, 4),
        )

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Run batch inference on a DataFrame.

        Returns (is_anomaly_array, decision_scores_array).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted or loaded yet.")

        X, _ = self.feature_engineer.extract_batch_features(df, self.features)
        decision_scores = self.model.decision_function(X)
        is_anomaly = (decision_scores < self.decision_threshold)

        return is_anomaly, decision_scores

    def save_model(self, model_path: Path, metadata_path: Optional[Path] = None) -> None:
        """Save fitted model artifact and metadata."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

        meta_file = metadata_path or model_path.with_suffix(".json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=2)

    def load_model(self, model_path: Path, metadata_path: Optional[Path] = None) -> bool:
        """Load fitted model artifact and metadata."""
        if not model_path.exists():
            return False

        meta_file = metadata_path or model_path.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.training_metadata = json.load(f)
                self.version = self.training_metadata.get("version", "1.0.0")
                self.features = self.training_metadata.get("features", self.features)
                self.decision_threshold = float(self.training_metadata.get("decision_threshold", self.decision_threshold))
                medians = self.training_metadata.get("imputation_medians", {})
                self.feature_engineer.set_imputation_medians(medians)

        self.model = joblib.load(model_path)
        self.is_fitted = True
        return True
