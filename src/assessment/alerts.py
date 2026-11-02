"""Alert severity and event deduplication module for EDAS-IoT."""

from datetime import datetime
from typing import Dict, List, Optional
import uuid
from src.common.schemas import (
    AlertEvent,
    AnomalySeverity,
    ProcessedReading,
)


class AlertManager:
    """Manages alert events and deduplicates continuous anomaly episodes."""

    def __init__(self):
        # Active alert events mapped by device_id: Dict[device_id, AlertEvent]
        self._active_events: Dict[str, AlertEvent] = {}
        # History of all alerts (active and resolved)
        self.alert_history: List[AlertEvent] = []

    def _severity_rank(self, severity: AnomalySeverity) -> int:
        ranks = {
            AnomalySeverity.NORMAL: 0,
            AnomalySeverity.INFO: 1,
            AnomalySeverity.DATA_QUALITY: 2,
            AnomalySeverity.WARNING: 3,
            AnomalySeverity.CRITICAL: 4,
        }
        return ranks.get(severity, 0)

    def process_reading(self, reading: ProcessedReading) -> Optional[AlertEvent]:
        """Process a newly evaluated reading, updating active alert episodes or creating new ones.

        Returns an AlertEvent if an alert was triggered or updated, None if normal.
        """
        device_id = reading.raw.device_id
        timestamp = reading.raw.timestamp
        severity = reading.severity

        # Determine all parameters involved in this anomaly
        params_involved = set(reading.statistical.flagged_parameters)
        if reading.validation.implausible_fields:
            params_involved.update(reading.validation.implausible_fields)
        if reading.validation.missing_fields:
            params_involved.update(reading.validation.missing_fields)
        if reading.violated_device_limits:
            for limit_str in reading.violated_device_limits:
                if "temperature" in limit_str.lower():
                    params_involved.add("temperature_c")
                if "humidity" in limit_str.lower():
                    params_involved.add("humidity_pct")
                if "pressure" in limit_str.lower():
                    params_involved.add("pressure_hpa")

        if reading.is_overall_anomaly or severity != AnomalySeverity.NORMAL:
            # Anomaly is currently active
            if device_id in self._active_events:
                # Update existing ongoing event
                event = self._active_events[device_id]
                event.last_detected = timestamp
                event.observation_count += 1

                # Escalate severity if current is higher
                if self._severity_rank(severity) > self._severity_rank(event.max_severity):
                    event.max_severity = severity

                # Update affected parameters
                updated_params = set(event.affected_parameters) | params_involved
                event.affected_parameters = sorted(list(updated_params))
                event.summary = reading.explanation
                return event
            else:
                # Create a new alert episode
                new_event = AlertEvent(
                    event_id=f"alert-{uuid.uuid4().hex[:8]}",
                    device_id=device_id,
                    first_detected=timestamp,
                    last_detected=timestamp,
                    observation_count=1,
                    max_severity=severity,
                    affected_parameters=sorted(list(params_involved)),
                    summary=reading.explanation,
                    status="ACTIVE",
                )
                self._active_events[device_id] = new_event
                self.alert_history.append(new_event)
                return new_event
        else:
            # Normal condition: resolve active event if one existed
            if device_id in self._active_events:
                event = self._active_events.pop(device_id)
                event.status = "RESOLVED"
                return event

            return None

    def get_active_alerts(self) -> List[AlertEvent]:
        """Return all currently active unresolved alert episodes."""
        return list(self._active_events.values())

    def get_all_alerts(self) -> List[AlertEvent]:
        """Return entire alert history."""
        return self.alert_history

    def reset_state(self) -> None:
        """Reset active events and history."""
        self._active_events.clear()
        self.alert_history.clear()
