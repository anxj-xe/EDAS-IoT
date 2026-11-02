"""Pydantic schemas and data models for EDAS-IoT."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"


class AnomalySeverity(str, Enum):
    NORMAL = "NORMAL"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DATA_QUALITY = "DATA_QUALITY"


class ReadinessStatus(str, Enum):
    READY = "READY"
    WARNING = "WARNING"
    NOT_READY = "NOT_READY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SENSOR_DATA_INVALID = "SENSOR_DATA_INVALID"


class AnomalyType(str, Enum):
    NORMAL = "NORMAL"
    TEMPERATURE_SPIKE = "TEMPERATURE_SPIKE"
    TEMPERATURE_DROP = "TEMPERATURE_DROP"
    TEMPERATURE_DRIFT = "TEMPERATURE_DRIFT"
    HUMIDITY_OUT_OF_RANGE = "HUMIDITY_OUT_OF_RANGE"
    PRESSURE_SPIKE = "PRESSURE_SPIKE"
    STUCK_SENSOR = "STUCK_SENSOR"
    MISSING_DATA = "MISSING_DATA"
    INVALID_MEASUREMENT = "INVALID_MEASUREMENT"
    MULTIVARIATE_ANOMALY = "MULTIVARIATE_ANOMALY"
    UNKNOWN = "UNKNOWN"


class RawSensorReading(BaseModel):
    """Raw sensor input schema conforming to ESP32 / CSV payload."""
    device_id: str = Field(..., description="Unique identifier for the sensor node")
    timestamp: str = Field(..., description="ISO 8601 timestamp string (UTC preferred)")
    temperature_c: Optional[float] = Field(None, description="Ambient temperature in Celsius")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity percentage (0-100%)")
    pressure_hpa: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    vibration: Optional[float] = Field(None, description="Vibration / acceleration metric")
    battery_voltage: Optional[float] = Field(None, description="Supply/battery voltage")
    sensor_status: Optional[str] = Field("OK", description="Hardware status code from sensor")
    sequence_number: Optional[int] = Field(None, description="Monotonically increasing sequence number")

    # Extra fields allowed to preserve unmodeled sensor telemetry without failing
    model_config = {
        "extra": "allow"
    }

    @field_validator("timestamp")
    def validate_timestamp_format(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Timestamp must be a non-empty string.")
        # Check standard ISO 8601 parsing
        try:
            # Replace Z with +00:00 for fromisoformat compatibility
            iso_str = v.replace("Z", "+00:00")
            datetime.fromisoformat(iso_str)
        except Exception as e:
            raise ValueError(f"Invalid ISO 8601 timestamp '{v}': {e}")
        return v


class ValidationResult(BaseModel):
    """Validation output for a sensor observation."""
    status: ValidationStatus = ValidationStatus.VALID
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    implausible_fields: List[str] = Field(default_factory=list)


class StatisticalResult(BaseModel):
    """Results from statistical anomaly detection."""
    is_anomaly: bool = False
    z_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    rolling_deviations: Dict[str, Optional[float]] = Field(default_factory=dict)
    flagged_parameters: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


class MLModelResult(BaseModel):
    """Output from Isolation Forest anomaly detection."""
    model_name: str = "isolation_forest"
    is_anomaly: bool = False
    decision_score: float = 0.0
    anomaly_score: float = 0.5


class DeviceProfile(BaseModel):
    """Configurable target IoT device specification."""
    name: str
    description: Optional[str] = ""
    min_temperature_c: Optional[float] = None
    max_temperature_c: Optional[float] = None
    min_humidity_pct: Optional[float] = None
    max_humidity_pct: Optional[float] = None
    min_pressure_hpa: Optional[float] = None
    max_pressure_hpa: Optional[float] = None
    required_parameters: List[str] = Field(default_factory=lambda: ["temperature_c", "humidity_pct"])
    safety_margins: Dict[str, float] = Field(default_factory=dict)


class ReadinessAssessment(BaseModel):
    """Deployment readiness evaluation based on observation window."""
    device_profile: str
    status: ReadinessStatus
    reasons: List[str] = Field(default_factory=list)
    window_size: int = 0
    valid_sample_count: int = 0
    out_of_bounds_count: int = 0
    in_bounds_ratio: float = 0.0
    violated_parameters: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    assessment_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProcessedReading(BaseModel):
    """Full aggregate record for storage, dashboard, and alerts."""
    raw: RawSensorReading
    validation: ValidationResult
    statistical: StatisticalResult
    ml: MLModelResult
    is_overall_anomaly: bool = False
    severity: AnomalySeverity = AnomalySeverity.NORMAL
    explanation: str = "Conditions appear normal."
    observed_evidence: List[str] = Field(default_factory=list)
    suspected_cause: Optional[str] = None
    violated_device_limits: List[str] = Field(default_factory=list)
    ground_truth_label: Optional[str] = None


class AlertEvent(BaseModel):
    """Deduplicated anomaly event spanning multiple observations."""
    event_id: str
    device_id: str
    first_detected: str
    last_detected: str
    observation_count: int = 1
    max_severity: AnomalySeverity = AnomalySeverity.INFO
    affected_parameters: List[str] = Field(default_factory=list)
    summary: str
    status: str = "ACTIVE" # ACTIVE or RESOLVED
