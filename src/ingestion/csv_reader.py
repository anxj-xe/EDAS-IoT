"""CSV Ingestion module (Mode A) for EDAS-IoT.

Loads historical sensor readings from CSV files, validates schemas,
runs the complete detection and readiness pipeline, and exports results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from src.common.config import get_config
from src.common.schemas import (
    DeviceProfile,
    ProcessedReading,
    ReadinessAssessment,
)
from src.detection.baseline import BaselineCalibrator, BaselineModel
from src.detection.isolation_forest import IsolationForestDetector
from src.processing.pipeline import EDASPipeline


class CSVReader:
    """Ingests CSV sensor telemetry files and coordinates pipeline processing."""

    REQUIRED_COLUMNS = ["timestamp", "device_id"]
    EXPECTED_COLUMNS = ["temperature_c", "humidity_pct", "pressure_hpa"]

    def __init__(
        self,
        pipeline: Optional[EDASPipeline] = None,
        device_profile: Optional[DeviceProfile] = None,
    ):
        self.config = get_config()
        self.pipeline = pipeline or self._init_default_pipeline(device_profile)

    def _init_default_pipeline(self, device_profile: Optional[DeviceProfile] = None) -> EDASPipeline:
        """Initialize pipeline with saved baseline and model if available."""
        models_dir = self.config.root_dir / "models"
        baseline_file = models_dir / "baseline.json"
        model_file = models_dir / "isolation_forest.joblib"

        baseline = BaselineModel()
        if baseline_file.exists():
            calibrator = BaselineCalibrator()
            baseline = calibrator.load_baseline(baseline_file)

        detector = IsolationForestDetector()
        if model_file.exists():
            detector.load_model(model_file)

        return EDASPipeline(
            baseline=baseline,
            ml_detector=detector,
            device_profile=device_profile,
        )

    def load_and_validate_csv(self, file_source: Union[str, Path, Any]) -> pd.DataFrame:
        """Read CSV file or buffer and check required column structure."""
        df = pd.read_csv(file_source)

        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise ValueError(f"CSV file is missing required columns: {missing_required}. Present columns: {df.columns.tolist()}")

        return df

    def run_csv_pipeline(
        self,
        file_source: Union[str, Path, Any],
        device_profile: Optional[DeviceProfile] = None,
    ) -> Tuple[pd.DataFrame, ReadinessAssessment, Dict[str, Any]]:
        """Execute complete pipeline on a CSV dataset.

        Returns:
            - DataFrame of processed readings with all flags, scores, and explanations
            - Final ReadinessAssessment
            - Summary statistics dictionary
        """
        df = self.load_and_validate_csv(file_source)

        if device_profile:
            self.pipeline.set_device_profile(device_profile)

        # Reset pipeline states for clean batch run
        self.pipeline.reset_state()

        processed_readings, readiness, alerts = self.pipeline.process_dataframe(df)

        # Convert processed readings to tabular DataFrame
        records = []
        for r in processed_readings:
            rec = {
                "timestamp": r.raw.timestamp,
                "device_id": r.raw.device_id,
                "temperature_c": r.raw.temperature_c,
                "humidity_pct": r.raw.humidity_pct,
                "pressure_hpa": r.raw.pressure_hpa,
                "is_valid": r.validation.is_valid,
                "validation_status": r.validation.status.value,
                "validation_errors": "; ".join(r.validation.errors) if r.validation.errors else "None",
                "is_statistical_anomaly": r.statistical.is_anomaly,
                "temp_z_score": r.statistical.z_scores.get("temperature_c"),
                "humidity_z_score": r.statistical.z_scores.get("humidity_pct"),
                "pressure_z_score": r.statistical.z_scores.get("pressure_hpa"),
                "is_ml_anomaly": r.ml.is_anomaly,
                "ml_decision_score": r.ml.decision_score,
                "is_overall_anomaly": r.is_overall_anomaly,
                "severity": r.severity.value,
                "explanation": r.explanation,
                "suspected_cause": r.suspected_cause or "None",
                "violated_device_limits": "; ".join(r.violated_device_limits) if r.violated_device_limits else "None",
                "ground_truth_label": r.ground_truth_label or "UNLABELLED",
            }
            records.append(rec)

        results_df = pd.DataFrame(records)

        summary = {
            "total_records": len(results_df),
            "valid_records": int(results_df["is_valid"].sum()),
            "invalid_records": int((~results_df["is_valid"]).sum()),
            "overall_anomalies": int(results_df["is_overall_anomaly"].sum()),
            "statistical_anomalies": int(results_df["is_statistical_anomaly"].sum()),
            "ml_anomalies": int(results_df["is_ml_anomaly"].sum()),
            "critical_alerts": int((results_df["severity"] == "CRITICAL").sum()),
            "warning_alerts": int((results_df["severity"] == "WARNING").sum()),
            "readiness_status": readiness.status.value,
            "readiness_compliance_pct": round(readiness.in_bounds_ratio * 100.0, 1),
            "active_alert_events": len(alerts),
        }

        return results_df, readiness, summary
