"""Unified detection and assessment pipeline for EDAS-IoT.

Shared by CSV mode, Simulated streaming mode, and ESP32 hardware mode.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.assessment.alerts import AlertManager
from src.assessment.explanations import AnomalyExplainer
from src.assessment.readiness import ReadinessAssessor
from src.common.config import get_config
from src.common.schemas import (
    AlertEvent,
    AnomalySeverity,
    DeviceProfile,
    ProcessedReading,
    RawSensorReading,
    ReadinessAssessment,
    ValidationStatus,
)
from src.detection.baseline import BaselineCalibrator, BaselineModel
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.statistical import StatisticalAnomalyDetector
from src.processing.validation import DataValidator


class EDASPipeline:
    """The central unified pipeline coordinating validation, detection, and assessment."""

    def __init__(
        self,
        baseline: Optional[BaselineModel] = None,
        ml_detector: Optional[IsolationForestDetector] = None,
        device_profile: Optional[DeviceProfile] = None,
    ):
        config = get_config()
        self.config = config

        # Components
        self.validator = DataValidator()
        self.baseline = baseline or BaselineModel()
        self.statistical_detector = StatisticalAnomalyDetector(baseline=self.baseline)
        self.ml_detector = ml_detector or IsolationForestDetector()

        default_profile = device_profile or config.get_device_profile(
            config.settings.get("assessment", {}).get("default_profile", "Standard_Indoor_Gateway")
        )
        self.readiness_assessor = ReadinessAssessor(device_profile=default_profile)
        self.explainer = AnomalyExplainer()
        self.alert_manager = AlertManager()

        # Update statistical detector if baseline is provided
        if baseline:
            self.statistical_detector.set_baseline(baseline)

    def set_baseline(self, baseline: BaselineModel) -> None:
        """Update active baseline across components."""
        self.baseline = baseline
        self.statistical_detector.set_baseline(baseline)

    def set_device_profile(self, profile: DeviceProfile) -> None:
        """Update active device profile across components."""
        self.readiness_assessor.set_device_profile(profile)

    def set_ml_detector(self, ml_detector: IsolationForestDetector) -> None:
        """Update ML detector."""
        self.ml_detector = ml_detector

    def process_reading(
        self, raw: RawSensorReading, ground_truth_label: Optional[str] = None
    ) -> Tuple[ProcessedReading, ReadinessAssessment, Optional[AlertEvent]]:
        """Process a single sensor reading through the complete pipeline.

        Returns (ProcessedReading, current ReadinessAssessment, AlertEvent if active/resolved).
        """
        # 1. Validation & Data Quality
        validation_res = self.validator.validate_reading(raw)

        # 2. Extract values for detection
        reading_dict = {
            "temperature_c": raw.temperature_c,
            "humidity_pct": raw.humidity_pct,
            "pressure_hpa": raw.pressure_hpa,
        }

        # 3. Statistical Anomaly Detection
        statistical_res = self.statistical_detector.detect(raw.device_id, reading_dict)

        # 4. ML Anomaly Detection (Isolation Forest)
        ml_res = self.ml_detector.predict_single(raw.device_id, reading_dict)

        # 5. Device Operating Limits Check
        # Create a preliminary processed reading for violation checking
        preliminary = ProcessedReading(
            raw=raw,
            validation=validation_res,
            statistical=statistical_res,
            ml=ml_res,
            ground_truth_label=ground_truth_label,
        )
        device_violations = self.readiness_assessor.check_reading_violations(preliminary)

        # 6. Root-Cause Explanation and Alert Severity
        explanation, evidence, suspected_cause, severity = self.explainer.explain(
            raw=raw,
            validation=validation_res,
            statistical=statistical_res,
            ml=ml_res,
            device_violations=device_violations,
        )

        # 7. Overall Anomaly Flag
        is_overall_anomaly = (
            (not validation_res.is_valid)
            or (validation_res.status == ValidationStatus.DATA_QUALITY_ISSUE)
            or statistical_res.is_anomaly
            or ml_res.is_anomaly
            or (len(device_violations) > 0)
        )

        # 8. Complete ProcessedReading Object
        processed = ProcessedReading(
            raw=raw,
            validation=validation_res,
            statistical=statistical_res,
            ml=ml_res,
            is_overall_anomaly=is_overall_anomaly,
            severity=severity,
            explanation=explanation,
            observed_evidence=evidence,
            suspected_cause=suspected_cause,
            violated_device_limits=device_violations,
            ground_truth_label=ground_truth_label,
        )

        # 9. Alert Deduplication / Update
        alert_event = self.alert_manager.process_reading(processed)

        # 10. Sliding Window Readiness Assessment
        readiness = self.readiness_assessor.add_reading_and_assess(processed)

        return processed, readiness, alert_event

    def process_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[List[ProcessedReading], ReadinessAssessment, List[AlertEvent]]:
        """Process a batch DataFrame of readings sequentially, maintaining sliding window states."""
        processed_records: List[ProcessedReading] = []
        latest_readiness: Optional[ReadinessAssessment] = None

        for _, row in df.iterrows():
            raw = RawSensorReading(
                device_id=str(row.get("device_id", "default_device")),
                timestamp=str(row.get("timestamp", "")),
                temperature_c=float(row["temperature_c"]) if pd.notna(row.get("temperature_c")) else None,
                humidity_pct=float(row["humidity_pct"]) if pd.notna(row.get("humidity_pct")) else None,
                pressure_hpa=float(row["pressure_hpa"]) if pd.notna(row.get("pressure_hpa")) else None,
                sensor_status=str(row.get("sensor_status", "OK")),
            )
            ground_truth = str(row.get("anomaly_type")) if pd.notna(row.get("anomaly_type")) else None

            p_reading, readiness, _ = self.process_reading(raw, ground_truth_label=ground_truth)
            processed_records.append(p_reading)
            latest_readiness = readiness

        all_alerts = self.alert_manager.get_all_alerts()
        # If no readings processed, return empty assessment
        if latest_readiness is None:
            latest_readiness = self.readiness_assessor.assess_window(
                "default", [], self.readiness_assessor.device_profile
            )

        return processed_records, latest_readiness, all_alerts

    def reset_state(self) -> None:
        """Reset internal buffers and history across all subcomponents."""
        self.validator.reset_state()
        self.statistical_detector.reset_state()
        self.readiness_assessor.reset_state()
        self.alert_manager.reset_state()
