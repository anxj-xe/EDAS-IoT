"""Model training and baseline calibration script for EDAS-IoT.

Calibrates the statistical baseline and trains the Isolation Forest ML model
strictly on normal environmental training data.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.common.config import get_config
from src.detection.baseline import BaselineCalibrator
from src.detection.isolation_forest import IsolationForestDetector


def train_and_calibrate(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    version: str = "1.0.0",
) -> None:
    """Run baseline calibration and fit Isolation Forest."""
    config = get_config()
    root = config.root_dir

    csv_file = data_path or (root / "data" / "samples" / "baseline_normal.csv")
    models_dir = output_dir or (root / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        raise FileNotFoundError(f"Training data not found at {csv_file}. Please run scripts/generate_data.py first.")

    print(f"Loading normal training data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records for calibration and training.")

    # 1. Baseline Calibration
    print("\n--- Establishing Environmental Baseline ---")
    calibrator = BaselineCalibrator(min_samples=20)
    baseline = calibrator.fit_from_dataframe(df, dataset_name=csv_file.name)

    baseline_file = models_dir / "baseline.json"
    calibrator.save_baseline(baseline_file)
    print(f"Baseline status: {baseline.status} (Sample count: {baseline.sample_count})")
    for param, stats in baseline.parameters.items():
        print(f"  • {param}: mean={stats.mean:.2f}, std={stats.std:.2f}, min={stats.min:.2f}, max={stats.max:.2f}")
    print(f"Saved baseline metadata to: {baseline_file}")

    # 2. Train Isolation Forest
    print("\n--- Training Isolation Forest Model ---")
    det_cfg = config.get_detection_config().get("isolation_forest", {})
    features = det_cfg.get("features", ["temperature_c", "humidity_pct", "pressure_hpa"])
    contamination = det_cfg.get("contamination", 0.05)
    decision_threshold = det_cfg.get("decision_threshold", -0.05)
    n_estimators = det_cfg.get("n_estimators", 100)
    random_state = det_cfg.get("random_state", 42)

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        decision_threshold=decision_threshold,
        random_state=random_state,
        features=features,
    )
    meta = detector.fit(df, version=version)

    model_file = models_dir / f"isolation_forest_v{version}.joblib"
    meta_file = models_dir / f"isolation_forest_v{version}.json"
    detector.save_model(model_file, meta_file)

    # Also save as default active model
    default_model_file = models_dir / "isolation_forest.joblib"
    default_meta_file = models_dir / "isolation_forest.json"
    detector.save_model(default_model_file, default_meta_file)

    print(f"Fitted Isolation Forest on features: {detector.features}")
    print(f"Saved model artifact to: {model_file} and {default_model_file}")
    print(f"Saved model metadata to: {meta_file}")
    print("Model Training & Calibration successfully completed!")


if __name__ == "__main__":
    train_and_calibrate()
