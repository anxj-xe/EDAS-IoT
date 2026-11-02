"""Simulated streaming ingestion module (Mode B) for EDAS-IoT.

Streams sensor readings one-by-one with configurable playback delays
and live on-the-fly anomaly injection capabilities.
"""

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import pandas as pd
from src.common.config import get_config
from src.common.schemas import (
    DeviceProfile,
    ProcessedReading,
    RawSensorReading,
    ReadinessAssessment,
)
from src.detection.baseline import BaselineCalibrator, BaselineModel
from src.detection.isolation_forest import IsolationForestDetector
from src.processing.pipeline import EDASPipeline
from src.storage.repository import DatabaseRepository


class SensorSimulator:
    """Manages playback of environmental readings and live injection of anomalies."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        pipeline: Optional[EDASPipeline] = None,
        repository: Optional[DatabaseRepository] = None,
        delay_seconds: float = 1.0,
    ):
        config = get_config()
        self.config = config
        self.dataset_path = dataset_path or (config.root_dir / "data" / "samples" / "baseline_normal.csv")
        self.delay_seconds = delay_seconds

        self.repository = repository or DatabaseRepository()
        self.pipeline = pipeline or self._init_pipeline()

        self._df: Optional[pd.DataFrame] = None
        self._current_idx: int = 0
        self._is_running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Live injection overrides: active until cleared
        self._injected_override: Optional[Dict[str, Any]] = None
        self._injected_label: Optional[str] = None
        self._override_duration_remaining: int = 0

        self._callbacks: list[Callable[[ProcessedReading, ReadinessAssessment], None]] = []
        self._load_data()

    def _init_pipeline(self) -> EDASPipeline:
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

        return EDASPipeline(baseline=baseline, ml_detector=detector)

    def _load_data(self) -> None:
        path = Path(self.dataset_path)
        if path.exists():
            self._df = pd.read_csv(path)
        else:
            self._df = pd.DataFrame(columns=["timestamp", "device_id", "temperature_c", "humidity_pct", "pressure_hpa"])
        self._current_idx = 0

    def set_dataset(self, file_path: Union[str, Path]) -> None:
        """Switch source dataset."""
        self.dataset_path = Path(file_path)
        self._load_data()

    def set_delay(self, delay: float) -> None:
        """Update playback interval."""
        self.delay_seconds = max(0.05, delay)

    def register_callback(self, cb: Callable[[ProcessedReading, ReadinessAssessment], None]) -> None:
        """Register a callback invoked on each streamed reading."""
        self._callbacks.append(cb)

    def inject_anomaly(self, anomaly_type: str, duration_steps: int = 5) -> None:
        """Schedule a controlled live anomaly to be injected into future stream steps."""
        self._injected_label = anomaly_type
        self._override_duration_remaining = duration_steps

        if anomaly_type == "TEMPERATURE_SPIKE":
            self._injected_override = {"temperature_c_offset": +15.0}
        elif anomaly_type == "TEMPERATURE_DROP":
            self._injected_override = {"temperature_c_offset": -12.0}
        elif anomaly_type == "HUMIDITY_OUT_OF_RANGE":
            self._injected_override = {"humidity_pct_fixed": 98.0}
        elif anomaly_type == "PRESSURE_SPIKE":
            self._injected_override = {"pressure_hpa_offset": +40.0}
        elif anomaly_type == "STUCK_SENSOR":
            self._injected_override = {"stuck": True}
        elif anomaly_type == "MISSING_DATA":
            self._injected_override = {"temperature_c": None, "humidity_pct": None}
        elif anomaly_type == "INVALID_MEASUREMENT":
            self._injected_override = {"temperature_c": 160.0}
        elif anomaly_type == "MULTIVARIATE_ANOMALY":
            self._injected_override = {"temperature_c_fixed": 29.0, "humidity_pct_fixed": 75.0}
        else:
            self._injected_override = None

    def step(self) -> Optional[ProcessedReading]:
        """Process and advance one single reading."""
        if self._df is None or self._df.empty:
            return None

        # Loop around if reached end of dataset
        if self._current_idx >= len(self._df):
            self._current_idx = 0

        row = self._df.iloc[self._current_idx]
        self._current_idx += 1

        device_id = str(row.get("device_id", "sim_device_01"))
        now_ts = datetime.now(timezone.utc).isoformat()

        temp = float(row["temperature_c"]) if pd.notna(row.get("temperature_c")) else None
        hum = float(row["humidity_pct"]) if pd.notna(row.get("humidity_pct")) else None
        press = float(row["pressure_hpa"]) if pd.notna(row.get("pressure_hpa")) else None
        ground_truth = str(row.get("anomaly_type", "NORMAL"))

        # Apply live anomaly injection if active
        if self._override_duration_remaining > 0 and self._injected_override:
            ground_truth = self._injected_label or "INJECTED_ANOMALY"

            if "temperature_c_offset" in self._injected_override and temp is not None:
                temp += self._injected_override["temperature_c_offset"]
            if "temperature_c_fixed" in self._injected_override:
                temp = self._injected_override["temperature_c_fixed"]
            if "humidity_pct_fixed" in self._injected_override:
                hum = self._injected_override["humidity_pct_fixed"]
            if "pressure_hpa_offset" in self._injected_override and press is not None:
                press += self._injected_override["pressure_hpa_offset"]
            if "temperature_c" in self._injected_override:
                temp = self._injected_override["temperature_c"]
            if "humidity_pct" in self._injected_override:
                hum = self._injected_override["humidity_pct"]

            self._override_duration_remaining -= 1
            if self._override_duration_remaining <= 0:
                self._injected_override = None
                self._injected_label = None

        raw = RawSensorReading(
            device_id=device_id,
            timestamp=now_ts,
            temperature_c=temp,
            humidity_pct=hum,
            pressure_hpa=press,
            sensor_status="OK",
            sequence_number=self._current_idx,
        )

        processed, readiness, alert = self.pipeline.process_reading(raw, ground_truth_label=ground_truth)

        # Persist to database
        self.repository.insert_processed_reading(processed)
        self.repository.insert_readiness_assessment(readiness)
        if alert:
            self.repository.upsert_alert_event(alert)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(processed, readiness)
            except Exception:
                pass

        return processed

    def start_streaming(self) -> None:
        """Start background streaming thread."""
        if self._is_running:
            return

        self._is_running = True

        def _run_loop():
            while self._is_running:
                self.step()
                time.sleep(self.delay_seconds)

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()

    def stop_streaming(self) -> None:
        """Stop background streaming thread."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_streaming(self) -> bool:
        return self._is_running

    def get_progress(self) -> Dict[str, Any]:
        return {
            "current_index": self._current_idx,
            "total_records": len(self._df) if self._df is not None else 0,
            "is_running": self._is_running,
            "delay_seconds": self.delay_seconds,
            "active_injection": self._injected_label if self._override_duration_remaining > 0 else None,
        }
