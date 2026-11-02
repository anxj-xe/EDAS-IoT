"""Root-cause explanation module for EDAS-IoT.

Synthesizes observations from data validation, statistical tests, ML scores,
and device limit checks into transparent, human-readable explanations.
"""

from typing import List, Optional, Tuple
from src.common.schemas import (
    AnomalySeverity,
    MLModelResult,
    RawSensorReading,
    StatisticalResult,
    ValidationResult,
    ValidationStatus,
)


class AnomalyExplainer:
    """Generates transparent, structured explanations for environmental and sensor anomalies."""

    def explain(
        self,
        raw: RawSensorReading,
        validation: ValidationResult,
        statistical: StatisticalResult,
        ml: MLModelResult,
        device_violations: List[str],
    ) -> Tuple[str, List[str], Optional[str], AnomalySeverity]:
        """Synthesize observation into (explanation, observed_evidence, suspected_cause, severity)."""
        evidence: List[str] = []
        suspected_causes: List[str] = []
        severity = AnomalySeverity.NORMAL

        # 1. Check Data Validation Issues
        if not validation.is_valid:
            severity = AnomalySeverity.DATA_QUALITY
            evidence.extend(validation.errors)
            if validation.implausible_fields:
                suspected_causes.append(
                    f"Physical plausibility violation or corrupted telemetry for: {', '.join(validation.implausible_fields)}"
                )
            if validation.missing_fields:
                suspected_causes.append(f"Sensor communication dropout or unpopulated fields: {', '.join(validation.missing_fields)}")
            if any("disconnection" in err.lower() for err in validation.errors):
                suspected_causes.append("Sensor power failure, cable disconnection, or I2C bus lockup")

        elif validation.status == ValidationStatus.DATA_QUALITY_ISSUE:
            severity = AnomalySeverity.DATA_QUALITY
            evidence.extend(validation.warnings)
            if any("stuck" in w.lower() for w in validation.warnings):
                suspected_causes.append("Sensor ADC freeze, mechanical stall, or firmware buffering failure")
            else:
                suspected_causes.append("Telemetry quality degradation or intermittent sensor read failure")

        # 2. Check Device Limits Violations
        if device_violations:
            evidence.extend([f"Device Limit Breach: {v}" for v in device_violations])
            # Critical severity if device limits are breached
            if severity != AnomalySeverity.DATA_QUALITY:
                severity = AnomalySeverity.CRITICAL
            suspected_causes.append("Ambient conditions have breached configured operational limits for target device")

        # 3. Check Statistical Deviations
        if statistical.is_anomaly:
            if severity == AnomalySeverity.NORMAL:
                severity = AnomalySeverity.WARNING
            evidence.extend(statistical.reasons)

            flagged = statistical.flagged_parameters
            if "temperature_c" in flagged and "humidity_pct" in flagged:
                suspected_causes.append("Simultaneous thermal and humidity perturbation (e.g. ventilation failure, steam or heater event)")
            elif "temperature_c" in flagged:
                suspected_causes.append("Sudden thermal influx, HVAC outage, direct solar exposure, or heating element active")
            elif "humidity_pct" in flagged:
                suspected_causes.append("Moisture surge, water condensation, dehumidifier malfunction, or weather shift")
            elif "pressure_hpa" in flagged:
                suspected_causes.append("Rapid barometric pressure change, enclosed space pressurization, or weather front")

        # 4. Check ML Outlier (Isolation Forest)
        if ml.is_anomaly:
            ml_detail = f"Isolation Forest flagged observation as multivariate outlier (score: {ml.decision_score:.3f})."
            evidence.append(ml_detail)
            if severity == AnomalySeverity.NORMAL:
                # If statistical tests didn't flag, but ML did, it's an unusual multivariate pattern
                severity = AnomalySeverity.INFO
                suspected_causes.append(
                    "Unusual multivariate combination: values are within individual bounds but their combined joint pattern is statistically rare."
                )

        # 5. Formulate final synthesized explanation
        if not evidence:
            explanation = "Environmental parameters are stable and within normal baseline bounds."
            suspected_cause = None
        else:
            primary_reason = evidence[0]
            if len(evidence) == 1:
                explanation = f"Flagged Condition: {primary_reason}"
            else:
                explanation = f"Multiple anomalies detected: {primary_reason} (+{len(evidence)-1} other factors)."

            suspected_cause = "; ".join(suspected_causes) if suspected_causes else "Unknown environmental shift."
            suspected_cause += " (Note: Final root cause requires on-site physical verification.)"

        return explanation, evidence, suspected_cause, severity
