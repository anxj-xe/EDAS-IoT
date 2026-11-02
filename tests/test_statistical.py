"""Unit tests for statistical anomaly detection in EDAS-IoT."""

import pytest
from src.detection.baseline import BaselineModel, ParameterStats
from src.detection.statistical import StatisticalAnomalyDetector


@pytest.fixture
def calibrated_baseline():
    """Create a calibrated baseline with known mean and std."""
    return BaselineModel(
        status="READY",
        sample_count=200,
        parameters={
            "temperature_c": ParameterStats(
                count=200, mean=20.0, std=2.0, min=15.0, max=25.0, median=20.0, q25=18.5, q75=21.5
            ),
            "humidity_pct": ParameterStats(
                count=200, mean=50.0, std=5.0, min=35.0, max=65.0, median=50.0, q25=46.0, q75=54.0
            ),
            "pressure_hpa": ParameterStats(
                count=200, mean=1013.0, std=1.0, min=1010.0, max=1016.0, median=1013.0, q25=1012.0, q75=1014.0
            ),
        },
    )


def test_normal_reading_not_flagged(calibrated_baseline):
    """Case 5: A normal reading close to baseline mean is not flagged."""
    detector = StatisticalAnomalyDetector(baseline=calibrated_baseline, z_score_threshold=3.0)
    readings = {"temperature_c": 21.0, "humidity_pct": 52.0, "pressure_hpa": 1013.5}

    result = detector.detect("node_01", readings)
    assert result.is_anomaly is False
    assert len(result.flagged_parameters) == 0
    assert abs(result.z_scores["temperature_c"]) < 3.0


def test_statistical_spike_detected(calibrated_baseline):
    """Case 4: A temperature spike (> 3 sigma) is flagged."""
    detector = StatisticalAnomalyDetector(baseline=calibrated_baseline, z_score_threshold=3.0)
    # Mean is 20.0, std is 2.0. Temp 30.0 -> z = (30 - 20) / 2 = 5.0
    readings = {"temperature_c": 30.0, "humidity_pct": 50.0, "pressure_hpa": 1013.0}

    result = detector.detect("node_01", readings)
    assert result.is_anomaly is True
    assert "temperature_c" in result.flagged_parameters
    assert result.z_scores["temperature_c"] >= 4.9
    assert any("Z-score" in r for r in result.reasons)


def test_zero_standard_deviation_safety():
    """Case 6: Zero standard deviation does not cause division by zero or infinite scores."""
    zero_std_baseline = BaselineModel(
        status="READY",
        sample_count=100,
        parameters={
            "temperature_c": ParameterStats(
                count=100, mean=25.0, std=0.0, min=25.0, max=25.0, median=25.0, q25=25.0, q75=25.0
            )
        },
        epsilon=1e-6,
    )
    detector = StatisticalAnomalyDetector(baseline=zero_std_baseline, epsilon=1e-6)
    readings = {"temperature_c": 25.0}

    result = detector.detect("node_01", readings)
    assert result.z_scores["temperature_c"] == 0.0

    # Slight deviation from perfectly constant baseline
    dev_readings = {"temperature_c": 25.5}
    dev_result = detector.detect("node_01", dev_readings)
    # Should not raise ZeroDivisionError
    assert isinstance(dev_result.z_scores["temperature_c"], float)
