"""Environmental readiness assessment module for EDAS-IoT.

Evaluates whether measured environmental conditions within an observation window
satisfy the operating constraints and safety margins of a target IoT device.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from src.common.schemas import (
    DeviceProfile,
    ProcessedReading,
    ReadinessAssessment,
    ReadinessStatus,
    ValidationStatus,
)


class ReadinessAssessor:
    """Evaluates environmental suitability for specific IoT device profiles."""

    def __init__(
        self,
        device_profile: Optional[DeviceProfile] = None,
        window_size: int = 20,
        min_samples: int = 10,
        readiness_threshold_pct: float = 90.0,
    ):
        self.device_profile = device_profile
        self.window_size = window_size
        self.min_samples = min_samples
        self.readiness_threshold_pct = readiness_threshold_pct

        # Per-device observation window: Dict[device_id, deque[ProcessedReading]]
        self._window_buffers: Dict[str, deque] = {}

    def set_device_profile(self, profile: DeviceProfile) -> None:
        """Update active target device profile."""
        self.device_profile = profile

    def assess_window(
        self, device_id: str, readings: List[ProcessedReading], profile: Optional[DeviceProfile] = None
    ) -> ReadinessAssessment:
        """Assess environmental readiness across a batch or window of processed readings."""
        target_profile = profile or self.device_profile
        now_iso = datetime.now(timezone.utc).isoformat()

        if target_profile is None:
            return ReadinessAssessment(
                device_profile="None",
                status=ReadinessStatus.INSUFFICIENT_DATA,
                reasons=["No target IoT device profile configured for assessment."],
                window_size=len(readings),
                valid_sample_count=0,
                out_of_bounds_count=0,
                in_bounds_ratio=0.0,
                violated_parameters=[],
                warnings=["Device profile missing."],
                assessment_timestamp=now_iso,
            )

        total_samples = len(readings)
        if total_samples < self.min_samples:
            return ReadinessAssessment(
                device_profile=target_profile.name,
                status=ReadinessStatus.INSUFFICIENT_DATA,
                reasons=[
                    f"Insufficient observations: received {total_samples} samples, but minimum required window is {self.min_samples}."
                ],
                window_size=total_samples,
                valid_sample_count=total_samples,
                out_of_bounds_count=0,
                in_bounds_ratio=0.0,
                violated_parameters=[],
                warnings=[],
                assessment_timestamp=now_iso,
            )

        # Check data quality across window
        invalid_count = sum(1 for r in readings if not r.validation.is_valid)
        if invalid_count / total_samples > 0.4:
            return ReadinessAssessment(
                device_profile=target_profile.name,
                status=ReadinessStatus.SENSOR_DATA_INVALID,
                reasons=[
                    f"Sensor data quality is too poor for trustworthy assessment ({invalid_count}/{total_samples} samples invalid)."
                ],
                window_size=total_samples,
                valid_sample_count=total_samples - invalid_count,
                out_of_bounds_count=0,
                in_bounds_ratio=0.0,
                violated_parameters=[],
                warnings=["Data quality prevents confident deployment evaluation."],
                assessment_timestamp=now_iso,
            )

        # Check required parameters presence
        for req_param in target_profile.required_parameters:
            param_present = any(
                getattr(r.raw, req_param, None) is not None for r in readings if r.validation.is_valid
            )
            if not param_present:
                return ReadinessAssessment(
                    device_profile=target_profile.name,
                    status=ReadinessStatus.NOT_READY,
                    reasons=[
                        f"Target device requires parameter '{req_param}', but it is completely absent in valid observations."
                    ],
                    window_size=total_samples,
                    valid_sample_count=total_samples - invalid_count,
                    out_of_bounds_count=total_samples,
                    in_bounds_ratio=0.0,
                    violated_parameters=[req_param],
                    warnings=[],
                    assessment_timestamp=now_iso,
                )

        # Evaluate limit violations and near-limit conditions
        out_of_bounds_count = 0
        violated_params_set = set()
        warning_params_set = set()
        reasons = []
        warnings = []

        temp_min = target_profile.min_temperature_c
        temp_max = target_profile.max_temperature_c
        temp_margin = target_profile.safety_margins.get("temperature_c", 2.0)

        hum_min = target_profile.min_humidity_pct
        hum_max = target_profile.max_humidity_pct
        hum_margin = target_profile.safety_margins.get("humidity_pct", 5.0)

        press_min = target_profile.min_pressure_hpa
        press_max = target_profile.max_pressure_hpa

        valid_readings = [r for r in readings if r.validation.is_valid]
        for r in valid_readings:
            reading_out_of_bounds = False

            # Temperature checks
            t = r.raw.temperature_c
            if t is not None:
                if (temp_min is not None and t < temp_min) or (temp_max is not None and t > temp_max):
                    reading_out_of_bounds = True
                    violated_params_set.add("temperature_c")
                elif (temp_min is not None and t < (temp_min + temp_margin)) or (
                    temp_max is not None and t > (temp_max - temp_margin)
                ):
                    warning_params_set.add("temperature_c")

            # Humidity checks
            h = r.raw.humidity_pct
            if h is not None:
                if (hum_min is not None and h < hum_min) or (hum_max is not None and h > hum_max):
                    reading_out_of_bounds = True
                    violated_params_set.add("humidity_pct")
                elif (hum_min is not None and h < (hum_min + hum_margin)) or (
                    hum_max is not None and h > (hum_max - hum_margin)
                ):
                    warning_params_set.add("humidity_pct")

            # Pressure checks
            p = r.raw.pressure_hpa
            if p is not None:
                if (press_min is not None and p < press_min) or (press_max is not None and p > press_max):
                    reading_out_of_bounds = True
                    violated_params_set.add("pressure_hpa")

            if reading_out_of_bounds:
                out_of_bounds_count += 1

        valid_count = len(valid_readings)
        in_bounds_count = valid_count - out_of_bounds_count
        in_bounds_ratio = round(in_bounds_count / max(1, valid_count), 4)
        in_bounds_pct = in_bounds_ratio * 100.0

        violated_list = sorted(list(violated_params_set))

        # Determine Readiness Decision
        if out_of_bounds_count > 0:
            if in_bounds_pct < self.readiness_threshold_pct:
                status = ReadinessStatus.NOT_READY
                reasons.append(
                    f"Conditions violate device operating limits in {out_of_bounds_count}/{valid_count} observations ({100.0 - in_bounds_pct:.1f}% violation rate)."
                )
                for vp in violated_list:
                    reasons.append(f"Parameter '{vp}' breached configured limits.")
            else:
                # Intermittent minor violation
                status = ReadinessStatus.WARNING
                reasons.append(
                    f"Occasional limit violation observed ({out_of_bounds_count}/{valid_count} samples), but compliance is {in_bounds_pct:.1f}%."
                )
        elif len(warning_params_set) > 0:
            status = ReadinessStatus.WARNING
            params_str = ", ".join(sorted(list(warning_params_set)))
            warnings.append(f"Operating conditions are near limit safety margins for: {params_str}.")
            reasons.append(f"Conditions are within operating limits but approach safety buffer for: {params_str}.")
        else:
            status = ReadinessStatus.READY
            reasons.append(
                f"Environment is stable and within all configured limits across {valid_count} observations (100% compliance)."
            )

        return ReadinessAssessment(
            device_profile=target_profile.name,
            status=status,
            reasons=reasons,
            window_size=total_samples,
            valid_sample_count=valid_count,
            out_of_bounds_count=out_of_bounds_count,
            in_bounds_ratio=in_bounds_ratio,
            violated_parameters=violated_list,
            warnings=warnings,
            assessment_timestamp=now_iso,
        )

    def check_reading_violations(
        self, reading: ProcessedReading, profile: Optional[DeviceProfile] = None
    ) -> List[str]:
        """Check if a single reading breaches target device limits."""
        target_profile = profile or self.device_profile
        if not target_profile:
            return []

        violations = []
        raw = reading.raw

        t = raw.temperature_c
        if t is not None:
            if target_profile.min_temperature_c is not None and t < target_profile.min_temperature_c:
                violations.append(f"Temperature {t:.1f}°C is below minimum limit ({target_profile.min_temperature_c}°C)")
            if target_profile.max_temperature_c is not None and t > target_profile.max_temperature_c:
                violations.append(f"Temperature {t:.1f}°C is above maximum limit ({target_profile.max_temperature_c}°C)")

        h = raw.humidity_pct
        if h is not None:
            if target_profile.min_humidity_pct is not None and h < target_profile.min_humidity_pct:
                violations.append(f"Humidity {h:.1f}% is below minimum limit ({target_profile.min_humidity_pct}%)")
            if target_profile.max_humidity_pct is not None and h > target_profile.max_humidity_pct:
                violations.append(f"Humidity {h:.1f}% is above maximum limit ({target_profile.max_humidity_pct}%)")

        p = raw.pressure_hpa
        if p is not None:
            if target_profile.min_pressure_hpa is not None and p < target_profile.min_pressure_hpa:
                violations.append(f"Pressure {p:.1f} hPa is below minimum limit ({target_profile.min_pressure_hpa} hPa)")
            if target_profile.max_pressure_hpa is not None and p > target_profile.max_pressure_hpa:
                violations.append(f"Pressure {p:.1f} hPa is above maximum limit ({target_profile.max_pressure_hpa} hPa)")

        return violations

    def add_reading_and_assess(
        self, reading: ProcessedReading, profile: Optional[DeviceProfile] = None
    ) -> ReadinessAssessment:
        """Stream interface: add reading to sliding window buffer and assess."""
        device_id = reading.raw.device_id
        if device_id not in self._window_buffers:
            self._window_buffers[device_id] = deque(maxlen=self.window_size)

        self._window_buffers[device_id].append(reading)
        window_list = list(self._window_buffers[device_id])
        return self.assess_window(device_id, window_list, profile)

    def reset_state(self, device_id: Optional[str] = None) -> None:
        """Reset internal sliding window buffers."""
        if device_id:
            self._window_buffers.pop(device_id, None)
        else:
            self._window_buffers.clear()
