"""Unit tests for Isolation Forest machine learning model in EDAS-IoT."""

import numpy as np
import pandas as pd
import pytest
from src.detection.isolation_forest import IsolationForestDetector


@pytest.fixture
def synthetic_training_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "temperature_c": 20.0 + np.random.normal(0, 1.0, n),
        "humidity_pct": 50.0 + np.random.normal(0, 3.0, n),
        "pressure_hpa": 1013.0 + np.random.normal(0, 0.5, n),
    })


def test_isolation_forest_train_and_inference(synthetic_training_df):
    """Case 7: Isolation Forest training and inference work."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    meta = detector.fit(synthetic_training_df)

    assert detector.is_fitted is True
    assert "version" in meta
    assert meta["n_samples"] == 100

    # Normal sample inference
    normal_reading = {"temperature_c": 20.1, "humidity_pct": 50.5, "pressure_hpa": 1013.2}
    res_normal = detector.predict_single("node_01", normal_reading)
    assert res_normal.model_name == "isolation_forest"
    assert res_normal.decision_score > -0.15 # Typical normal decision score

    # Highly anomalous multivariate sample inference
    extreme_reading = {"temperature_c": 45.0, "humidity_pct": 95.0, "pressure_hpa": 980.0}
    res_extreme = detector.predict_single("node_01", extreme_reading)
    assert res_extreme.is_anomaly is True
    assert res_extreme.decision_score < 0.0


def test_model_save_and_reload(synthetic_training_df, tmp_path):
    """Case 8: The model artifact can be saved and reloaded."""
    detector = IsolationForestDetector(n_estimators=30, random_state=42)
    detector.fit(synthetic_training_df)

    model_file = tmp_path / "test_iso_forest.joblib"
    meta_file = tmp_path / "test_iso_forest.json"

    detector.save_model(model_file, meta_file)
    assert model_file.exists()
    assert meta_file.exists()

    # Load into fresh detector
    reloaded_detector = IsolationForestDetector()
    success = reloaded_detector.load_model(model_file, meta_file)
    assert success is True
    assert reloaded_detector.is_fitted is True
    assert reloaded_detector.features == detector.features

    # Inference with reloaded model
    test_reading = {"temperature_c": 20.0, "humidity_pct": 50.0, "pressure_hpa": 1013.0}
    res_orig = detector.predict_single("node_01", test_reading)
    res_loaded = reloaded_detector.predict_single("node_01", test_reading)

    assert res_orig.decision_score == res_loaded.decision_score
