"""Unit tests for data validation, edge cases, and quality checks in EDAS-IoT."""

import pytest
from src.common.schemas import RawSensorReading, ValidationStatus
from src.processing.validation import DataValidator


@pytest.fixture
def validator():
    return DataValidator(
        physical_limits={
            "temperature_c": {"min": -40.0, "max": 85.0},
            "humidity_pct": {"min": 0.0, "max": 100.0},
            "pressure_hpa": {"min": 300.0, "max": 1100.0},
        },
        stuck_window_size=4,
    )


def test_valid_normal_reading(validator):
    """Case 1: A valid normal reading passes validation."""
    reading = RawSensorReading(
        device_id="node_01",
        timestamp="2026-10-13T10:00:00Z",
        temperature_c=22.5,
        humidity_pct=50.0,
        pressure_hpa=1013.25,
    )
    result = validator.validate_reading(reading)
    assert result.is_valid is True
    assert result.status == ValidationStatus.VALID
    assert len(result.errors) == 0


def test_missing_required_device_id(validator):
    """Case 2: Missing or whitespace device_id produces an error."""
    reading = RawSensorReading(
        device_id="   ",
        timestamp="2026-10-13T10:00:00Z",
        temperature_c=22.5,
    )
    result = validator.validate_reading(reading)
    assert result.is_valid is False
    assert any("device_id" in err for err in result.errors)


def test_invalid_timestamp_rejected():
    """Case 3: Malformed timestamps fail pydantic schema validation."""
    with pytest.raises(Exception):
        RawSensorReading(
            device_id="node_01",
            timestamp="not-a-timestamp",
            temperature_c=20.0,
        )


def test_physically_implausible_limits(validator):
    """Physically impossible values (e.g. 150°C, 120% RH) fail validation."""
    reading = RawSensorReading(
        device_id="node_01",
        timestamp="2026-10-13T10:00:00Z",
        temperature_c=150.0, # Too hot for Earth ambient
        humidity_pct=110.0,  # Cannot exceed 100%
        pressure_hpa=1013.0,
    )
    result = validator.validate_reading(reading)
    assert result.is_valid is False
    assert result.status == ValidationStatus.INVALID
    assert "temperature_c" in result.implausible_fields
    assert "humidity_pct" in result.implausible_fields


def test_stuck_sensor_detection(validator):
    """Identical repeated readings over stuck_window_size trigger DATA_QUALITY_ISSUE."""
    for i in range(3):
        r = RawSensorReading(
            device_id="node_01",
            timestamp=f"2026-10-13T10:0{i}:00Z",
            temperature_c=25.0,
            humidity_pct=60.0,
            pressure_hpa=1013.0,
        )
        res = validator.validate_reading(r)
        assert res.status == ValidationStatus.VALID

    # 4th identical reading triggers stuck sensor warning
    r4 = RawSensorReading(
        device_id="node_01",
        timestamp="2026-10-13T10:04:00Z",
        temperature_c=25.0,
        humidity_pct=60.0,
        pressure_hpa=1013.0,
    )
    res4 = validator.validate_reading(r4)
    assert res4.status == ValidationStatus.DATA_QUALITY_ISSUE
    assert any("stuck sensor" in w.lower() for w in res4.warnings)


def test_missing_optional_sensor_fields_handled_consistently(validator):
    """Case 9: Missing optional pressure does not invalidate temperature and humidity."""
    reading = RawSensorReading(
        device_id="node_01",
        timestamp="2026-10-13T10:00:00Z",
        temperature_c=21.0,
        humidity_pct=45.0,
        pressure_hpa=None, # e.g. DHT22 sensor without pressure
    )
    result = validator.validate_reading(reading)
    assert result.is_valid is True
    assert "pressure_hpa" in result.missing_fields
    assert len(result.errors) == 0


def test_duplicate_and_out_of_order_timestamps(validator):
    """Case 13: Duplicate and out-of-order timestamps generate warnings without silent dropping."""
    r1 = RawSensorReading(device_id="node_01", timestamp="2026-10-13T10:05:00Z", temperature_c=20.0)
    validator.validate_reading(r1)

    # Duplicate timestamp
    r_dup = RawSensorReading(device_id="node_01", timestamp="2026-10-13T10:05:00Z", temperature_c=21.0)
    res_dup = validator.validate_reading(r_dup)
    assert any("duplicate timestamp" in w.lower() for w in res_dup.warnings)

    # Out of order
    r_past = RawSensorReading(device_id="node_01", timestamp="2026-10-13T10:00:00Z", temperature_c=19.0)
    res_past = validator.validate_reading(r_past)
    assert any("out-of-order" in w.lower() for w in res_past.warnings)
