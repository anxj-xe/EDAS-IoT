"""Unit tests for environmental readiness assessment in EDAS-IoT."""

import pytest
from src.assessment.readiness import ReadinessAssessor
from src.common.schemas import (
    DeviceProfile,
    MLModelResult,
    ProcessedReading,
    RawSensorReading,
    ReadinessStatus,
    StatisticalResult,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture
def indoor_profile():
    return DeviceProfile(
        name="Standard_Indoor_Gateway",
        min_temperature_c=10.0,
        max_temperature_c=35.0,
        min_humidity_pct=20.0,
        max_humidity_pct=70.0,
        required_parameters=["temperature_c", "humidity_pct"],
        safety_margins={"temperature_c": 2.0, "humidity_pct": 5.0},
    )


def make_dummy_reading(temp: float, hum: float, is_anomaly: bool = False) -> ProcessedReading:
    raw = RawSensorReading(
        device_id="node_01",
        timestamp="2026-10-13T10:00:00Z",
        temperature_c=temp,
        humidity_pct=hum,
        pressure_hpa=1013.0,
    )
    val = ValidationResult(status=ValidationStatus.VALID, is_valid=True)
    stat = StatisticalResult(is_anomaly=is_anomaly)
    ml = MLModelResult(is_anomaly=is_anomaly)
    return ProcessedReading(
        raw=raw,
        validation=val,
        statistical=stat,
        ml=ml,
        is_overall_anomaly=is_anomaly,
    )


def test_insufficient_data_produces_insufficient_status(indoor_profile):
    """Case 12: Window with fewer than min_samples produces INSUFFICIENT_DATA."""
    assessor = ReadinessAssessor(device_profile=indoor_profile, min_samples=10)
    readings = [make_dummy_reading(22.0, 50.0) for _ in range(5)] # Only 5 samples

    assessment = assessor.assess_window("node_01", readings)
    assert assessment.status == ReadinessStatus.INSUFFICIENT_DATA
    assert any("Insufficient observations" in r for r in assessment.reasons)


def test_operating_limit_violation_produces_not_ready(indoor_profile):
    """Case 10: Sustained temperature exceeding device max operating limit produces NOT_READY."""
    assessor = ReadinessAssessor(device_profile=indoor_profile, min_samples=10, readiness_threshold_pct=90.0)
    # Target max temp is 35.0°C. Feed 42.0°C (hot environment)
    readings = [make_dummy_reading(42.0, 50.0) for _ in range(15)]

    assessment = assessor.assess_window("node_01", readings)
    assert assessment.status == ReadinessStatus.NOT_READY
    assert "temperature_c" in assessment.violated_parameters


def test_anomaly_alone_does_not_force_not_ready(indoor_profile):
    """Case 11: An anomaly alone does not produce NOT_READY if readings are within device limits."""
    assessor = ReadinessAssessor(device_profile=indoor_profile, min_samples=10)

    # 15 readings at 28.0°C (inside [10.0, 35.0] limits), but flagged as statistical anomaly relative to an 18°C baseline
    readings = [make_dummy_reading(28.0, 50.0, is_anomaly=True) for _ in range(15)]

    assessment = assessor.assess_window("node_01", readings)
    # Must NOT be NOT_READY, because device hardware limits [10, 35] are fully respected
    assert assessment.status != ReadinessStatus.NOT_READY
    assert assessment.status == ReadinessStatus.READY
    assert len(assessment.violated_parameters) == 0


def test_near_limit_triggers_warning(indoor_profile):
    """Conditions near limit (within safety margin) produce WARNING status."""
    assessor = ReadinessAssessor(device_profile=indoor_profile, min_samples=10)
    # Max is 35.0°C, margin is 2.0°C -> warning zone is [33.0, 35.0]
    readings = [make_dummy_reading(34.0, 50.0) for _ in range(15)]

    assessment = assessor.assess_window("node_01", readings)
    assert assessment.status == ReadinessStatus.WARNING
    assert "temperature_c" in assessment.warnings[0]
