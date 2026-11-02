"""Data validation and data quality checking module for EDAS-IoT."""

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from src.common.config import get_config
from src.common.schemas import (
    RawSensorReading,
    ValidationResult,
    ValidationStatus,
)


class DataValidator:
    """Validates raw sensor records and monitors data quality."""

    def __init__(self, physical_limits: Optional[Dict[str, Dict[str, float]]] = None, stuck_window_size: int = 5):
        if physical_limits is None:
            self.physical_limits = get_config().get_physical_limits()
        else:
            self.physical_limits = physical_limits
        self.stuck_window_size = stuck_window_size

        # Buffer for tracking repeated values per device: Dict[device_id, Dict[param, deque]]
        self._history: Dict[str, Dict[str, deque]] = {}
        # Monotonic timestamp tracker per device
        self._last_timestamp: Dict[str, datetime] = {}
        # Duplicate timestamp set per device (recent window)
        self._seen_timestamps: Dict[str, deque] = {}

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        try:
            iso_clean = ts_str.replace("Z", "+00:00")
            return datetime.fromisoformat(iso_clean)
        except Exception:
            return None

    def validate_reading(self, reading: RawSensorReading) -> ValidationResult:
        """Validate a single RawSensorReading against physical limits, consistency, and quality rules."""
        errors: List[str] = []
        warnings: List[str] = []
        missing_fields: List[str] = []
        implausible_fields: List[str] = []

        device_id = reading.device_id
        if not device_id or not device_id.strip():
            errors.append("Missing required field 'device_id'.")

        # 1. Timestamp validation
        dt = self._parse_timestamp(reading.timestamp)
        if dt is None:
            errors.append(f"Invalid timestamp format '{reading.timestamp}'.")
        else:
            # Check duplicate timestamp
            if device_id not in self._seen_timestamps:
                self._seen_timestamps[device_id] = deque(maxlen=200)

            if reading.timestamp in self._seen_timestamps[device_id]:
                warnings.append(f"Duplicate timestamp detected for device {device_id}: {reading.timestamp}")
            else:
                self._seen_timestamps[device_id].append(reading.timestamp)

            # Check chronological out-of-order
            if device_id in self._last_timestamp:
                if dt < self._last_timestamp[device_id]:
                    warnings.append(f"Out-of-order timestamp: {reading.timestamp} is earlier than previous reading.")
            self._last_timestamp[device_id] = dt

        # 2. Sensor hardware status check
        if reading.sensor_status and reading.sensor_status.upper() not in ["OK", "READY", "ONLINE"]:
            warnings.append(f"Hardware sensor status reports code: {reading.sensor_status}")

        # 3. Check physical limits and non-numeric values
        env_params = {
            "temperature_c": reading.temperature_c,
            "humidity_pct": reading.humidity_pct,
            "pressure_hpa": reading.pressure_hpa,
        }

        all_null = True
        for param, val in env_params.items():
            if val is None:
                missing_fields.append(param)
                continue

            all_null = False
            # Check if numeric
            if not isinstance(val, (int, float)):
                errors.append(f"Parameter '{param}' is non-numeric: {val}")
                implausible_fields.append(param)
                continue

            # Check physical limits
            limits = self.physical_limits.get(param)
            if limits:
                min_lim = limits.get("min", -float("inf"))
                max_lim = limits.get("max", float("inf"))
                if val < min_lim or val > max_lim:
                    errors.append(
                        f"Physically implausible value for '{param}': {val} is outside physical bounds [{min_lim}, {max_lim}]."
                    )
                    implausible_fields.append(param)

        if all_null:
            errors.append("Sensor disconnection or complete measurement loss: all environmental readings are missing.")

        # 4. Stuck sensor detection (repeated unchanged readings)
        if device_id not in self._history:
            self._history[device_id] = {
                "temperature_c": deque(maxlen=self.stuck_window_size),
                "humidity_pct": deque(maxlen=self.stuck_window_size),
                "pressure_hpa": deque(maxlen=self.stuck_window_size),
            }

        stuck_detected = False
        for param, val in env_params.items():
            buf = self._history[device_id][param]
            if val is not None:
                buf.append(val)
                if len(buf) == self.stuck_window_size:
                    # Check if all values in buffer are essentially identical
                    first_val = buf[0]
                    if all(abs(x - first_val) < 1e-5 for x in buf):
                        warnings.append(
                            f"Stuck sensor detected for '{param}': constant value {first_val} across {self.stuck_window_size} consecutive readings."
                        )
                        stuck_detected = True

        # Determine overall validation status
        if errors:
            status = ValidationStatus.INVALID
            is_valid = False
        elif stuck_detected or len(missing_fields) > 1:
            status = ValidationStatus.DATA_QUALITY_ISSUE
            is_valid = True
        else:
            status = ValidationStatus.VALID
            is_valid = True

        return ValidationResult(
            status=status,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            implausible_fields=implausible_fields,
        )

    def reset_state(self, device_id: Optional[str] = None) -> None:
        """Reset internal history and tracking for testing or restart."""
        if device_id:
            self._history.pop(device_id, None)
            self._last_timestamp.pop(device_id, None)
            self._seen_timestamps.pop(device_id, None)
        else:
            self._history.clear()
            self._last_timestamp.clear()
            self._seen_timestamps.clear()
