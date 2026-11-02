"""Controlled synthetic data generator for EDAS-IoT.

Generates realistic environmental data with diurnal cycles, sensor noise,
and 10 controlled anomaly types with ground truth labels.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_environmental_dataset(
    n_samples: int = 500,
    device_id: str = "environment_node_01",
    start_time: Optional[datetime] = None,
    interval_seconds: int = 60,
    inject_anomalies: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate time series data for temperature, humidity, and atmospheric pressure.

    If inject_anomalies is False: creates normal environmental baseline data.
    If inject_anomalies is True: introduces controlled anomalies with ground-truth labels.
    """
    np.random.seed(random_seed)
    if start_time is None:
        start_time = datetime(2026, 10, 13, 8, 0, 0, tzinfo=timezone.utc)

    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(n_samples)]

    # Time in hours for diurnal sinusoidal variation
    time_hours = np.array([(i * interval_seconds) / 3600.0 for i in range(n_samples)])

    # Normal baseline signals with natural diurnal cycle and Gaussian sensor noise
    # Temperature: 22°C baseline + 3°C diurnal swing + 0.3°C noise
    temp = 22.0 + 3.0 * np.sin(2 * np.pi * (time_hours - 6) / 24.0) + np.random.normal(0, 0.3, n_samples)

    # Humidity: Inversely correlated with temperature (typical ambient physics): 50% baseline - 8% diurnal swing + 1% noise
    hum = 50.0 - 8.0 * np.sin(2 * np.pi * (time_hours - 6) / 24.0) + np.random.normal(0, 1.0, n_samples)

    # Pressure: 1013.25 hPa baseline + slight barometric drift + 0.2 hPa noise
    pressure = 1013.25 + 1.5 * np.cos(2 * np.pi * time_hours / 48.0) + np.random.normal(0, 0.2, n_samples)

    labels = ["NORMAL"] * n_samples
    sensor_status = ["OK"] * n_samples

    if inject_anomalies and n_samples >= 300:
        # Segment 1: Sudden Temperature Spike (samples 40:48)
        temp[40:48] += np.random.uniform(12.0, 16.0, 8)
        for idx in range(40, 48):
            labels[idx] = "TEMPERATURE_SPIKE"

        # Segment 2: Sudden Temperature Drop (samples 75:82)
        temp[75:82] -= np.random.uniform(10.0, 14.0, 7)
        for idx in range(75, 82):
            labels[idx] = "TEMPERATURE_DROP"

        # Segment 3: Gradual Temperature Drift (samples 110:135)
        drift = np.linspace(0.5, 12.0, 25)
        temp[110:135] += drift
        for idx in range(110, 135):
            labels[idx] = "TEMPERATURE_DRIFT"

        # Segment 4: Abnormally High Humidity (samples 160:175)
        hum[160:175] += np.random.uniform(32.0, 40.0, 15)
        for idx in range(160, 175):
            labels[idx] = "HUMIDITY_OUT_OF_RANGE"

        # Segment 5: Abnormally Low Humidity (samples 200:212)
        hum[200:212] = np.random.uniform(2.0, 6.0, 12)
        for idx in range(200, 212):
            labels[idx] = "HUMIDITY_OUT_OF_RANGE"

        # Segment 6: Pressure Spike / Drop (samples 230:238)
        pressure[230:234] += np.random.uniform(35.0, 50.0, 4)
        pressure[234:238] -= np.random.uniform(35.0, 50.0, 4)
        for idx in range(230, 238):
            labels[idx] = "PRESSURE_SPIKE"

        # Segment 7: Stuck Sensor (samples 255:265)
        stuck_temp = float(temp[254])
        stuck_hum = float(hum[254])
        stuck_press = float(pressure[254])
        for idx in range(255, 265):
            temp[idx] = stuck_temp
            hum[idx] = stuck_hum
            pressure[idx] = stuck_press
            labels[idx] = "STUCK_SENSOR"

        # Segment 8: Missing Readings / Dropout (samples 280:285)
        for idx in range(280, 285):
            labels[idx] = "MISSING_DATA"
            # Set to NaN or None
            temp[idx] = np.nan
            hum[idx] = np.nan

        # Segment 9: Physically Implausible Measurements (samples 295:299)
        temp[295] = 165.0  # Impossibly high temperature
        hum[296] = 135.0   # > 100% relative humidity
        pressure[297] = 150.0 # Way below barometric minimum
        temp[298] = -75.0  # Impossibly low temperature
        for idx in range(295, 299):
            labels[idx] = "INVALID_MEASUREMENT"
            sensor_status[idx] = "ERROR_OUT_OF_RANGE"

        # Segment 10: Multivariate Anomaly (samples 320:330)
        # Both temperature is somewhat elevated (e.g. 29°C) AND humidity is abnormally high (70%)
        # Neither individually breaches univariate 3-sigma thresholds, but their joint occurrence is extremely rare
        if n_samples >= 340:
            temp[320:330] = 28.5 + np.random.normal(0, 0.2, 10)
            hum[320:330] = 72.0 + np.random.normal(0, 0.5, 10)
            for idx in range(320, 330):
                labels[idx] = "MULTIVARIATE_ANOMALY"

    # Assemble into DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in timestamps],
        "device_id": device_id,
        "temperature_c": np.round(temp, 2),
        "humidity_pct": np.round(hum, 2),
        "pressure_hpa": np.round(pressure, 2),
        "sensor_status": sensor_status,
        "anomaly_type": labels,
    })

    return df


def main():
    """Generate sample datasets and save to data/samples/."""
    project_root = Path(__file__).resolve().parent.parent
    samples_dir = project_root / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Generating normal baseline dataset (1440 records, 24-hour cycle)...")
    baseline_df = generate_environmental_dataset(
        n_samples=1440,
        inject_anomalies=False,
        random_seed=42,
    )
    baseline_path = samples_dir / "baseline_normal.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"Saved: {baseline_path} ({len(baseline_df)} records)")

    print("Generating controlled evaluation dataset with 10 anomaly types (1440 records)...")
    eval_df = generate_environmental_dataset(
        n_samples=1440,
        inject_anomalies=True,
        random_seed=42,
    )
    eval_path = samples_dir / "evaluation_labeled.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved: {eval_path} ({len(eval_df)} records)")

    anomaly_counts = eval_df["anomaly_type"].value_counts()
    print("\nEvaluation Dataset Label Distribution:")
    for label, count in anomaly_counts.items():
        print(f"  - {label}: {count} samples")


if __name__ == "__main__":
    main()
