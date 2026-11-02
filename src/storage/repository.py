"""SQLite Persistence layer for EDAS-IoT.

Stores raw readings, validation results, statistical detections,
ML predictions, readiness evaluations, and alert events.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from src.common.config import get_config
from src.common.schemas import (
    AlertEvent,
    AnomalySeverity,
    ProcessedReading,
    RawSensorReading,
    ReadinessAssessment,
    ReadinessStatus,
    ValidationResult,
    ValidationStatus,
)


class DatabaseRepository:
    """Manages SQLite database connections, schema migrations, and queries."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            config = get_config()
            db_rel = config.settings.get("storage", {}).get("database_path", "data/edas_iot.db")
            self.db_path = config.root_dir / db_rel
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Readings and Evaluations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    temperature_c REAL,
                    humidity_pct REAL,
                    pressure_hpa REAL,
                    sensor_status TEXT,
                    sequence_number INTEGER,
                    is_valid INTEGER NOT NULL,
                    validation_status TEXT NOT NULL,
                    validation_errors TEXT,
                    is_statistical_anomaly INTEGER NOT NULL,
                    temp_z_score REAL,
                    humidity_z_score REAL,
                    pressure_z_score REAL,
                    is_ml_anomaly INTEGER NOT NULL,
                    ml_decision_score REAL,
                    ml_anomaly_score REAL,
                    is_overall_anomaly INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT,
                    observed_evidence TEXT,
                    suspected_cause TEXT,
                    violated_device_limits TEXT,
                    ground_truth_label TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # 2. Readiness Assessments Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS readiness_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    window_size INTEGER NOT NULL,
                    valid_sample_count INTEGER NOT NULL,
                    out_of_bounds_count INTEGER NOT NULL,
                    in_bounds_ratio REAL NOT NULL,
                    reasons TEXT,
                    violated_parameters TEXT,
                    warnings TEXT,
                    assessment_timestamp TEXT NOT NULL
                )
            """)

            # 3. Alert Events Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    first_detected TEXT NOT NULL,
                    last_detected TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    max_severity TEXT NOT NULL,
                    affected_parameters TEXT,
                    summary TEXT,
                    status TEXT NOT NULL
                )
            """)

            # Indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_device_time ON readings(device_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_is_anomaly ON readings(is_overall_anomaly)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alert_events(status)")

            conn.commit()

    def insert_processed_reading(self, reading: ProcessedReading) -> int:
        """Insert a single processed reading."""
        raw = reading.raw
        val = reading.validation
        stat = reading.statistical
        ml = reading.ml

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO readings (
                    timestamp, device_id, temperature_c, humidity_pct, pressure_hpa,
                    sensor_status, sequence_number,
                    is_valid, validation_status, validation_errors,
                    is_statistical_anomaly, temp_z_score, humidity_z_score, pressure_z_score,
                    is_ml_anomaly, ml_decision_score, ml_anomaly_score,
                    is_overall_anomaly, severity, explanation,
                    observed_evidence, suspected_cause, violated_device_limits, ground_truth_label, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    raw.timestamp,
                    raw.device_id,
                    raw.temperature_c,
                    raw.humidity_pct,
                    raw.pressure_hpa,
                    raw.sensor_status,
                    raw.sequence_number,
                    1 if val.is_valid else 0,
                    val.status.value,
                    json.dumps(val.errors),
                    1 if stat.is_anomaly else 0,
                    stat.z_scores.get("temperature_c"),
                    stat.z_scores.get("humidity_pct"),
                    stat.z_scores.get("pressure_hpa"),
                    1 if ml.is_anomaly else 0,
                    ml.decision_score,
                    ml.anomaly_score,
                    1 if reading.is_overall_anomaly else 0,
                    reading.severity.value,
                    reading.explanation,
                    json.dumps(reading.observed_evidence),
                    reading.suspected_cause,
                    json.dumps(reading.violated_device_limits),
                    reading.ground_truth_label,
                    now_iso,
                ),
            )
            inserted_id = cursor.lastrowid
            conn.commit()
            return inserted_id

    def insert_readiness_assessment(self, assessment: ReadinessAssessment) -> int:
        """Insert a readiness evaluation record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO readiness_assessments (
                    device_profile, status, window_size, valid_sample_count,
                    out_of_bounds_count, in_bounds_ratio, reasons,
                    violated_parameters, warnings, assessment_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    assessment.device_profile,
                    assessment.status.value,
                    assessment.window_size,
                    assessment.valid_sample_count,
                    assessment.out_of_bounds_count,
                    assessment.in_bounds_ratio,
                    json.dumps(assessment.reasons),
                    json.dumps(assessment.violated_parameters),
                    json.dumps(assessment.warnings),
                    assessment.assessment_timestamp,
                ),
            )
            inserted_id = cursor.lastrowid
            conn.commit()
            return inserted_id

    def upsert_alert_event(self, event: AlertEvent) -> None:
        """Insert or update an alert event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alert_events (
                    event_id, device_id, first_detected, last_detected,
                    observation_count, max_severity, affected_parameters,
                    summary, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    last_detected = excluded.last_detected,
                    observation_count = excluded.observation_count,
                    max_severity = excluded.max_severity,
                    affected_parameters = excluded.affected_parameters,
                    summary = excluded.summary,
                    status = excluded.status
            """,
                (
                    event.event_id,
                    event.device_id,
                    event.first_detected,
                    event.last_detected,
                    event.observation_count,
                    event.max_severity.value,
                    json.dumps(event.affected_parameters),
                    event.summary,
                    event.status,
                ),
            )
            conn.commit()

    def get_latest_reading(self, device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent reading."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if device_id:
                cursor.execute(
                    "SELECT * FROM readings WHERE device_id = ? ORDER BY id DESC LIMIT 1",
                    (device_id,),
                )
            else:
                cursor.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_readings(self, limit: int = 100, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent readings for monitoring and charts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if device_id:
                cursor.execute(
                    "SELECT * FROM readings WHERE device_id = ? ORDER BY id DESC LIMIT ?",
                    (device_id, limit),
                )
            else:
                cursor.execute("SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            # Return in chronological ascending order for plotting
            results = [dict(r) for r in rows]
            results.reverse()
            return results

    def get_anomalies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve detected anomalies."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM readings WHERE is_overall_anomaly = 1 ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_latest_readiness(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent readiness assessment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM readiness_assessments ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["reasons"] = json.loads(res["reasons"]) if res.get("reasons") else []
                res["violated_parameters"] = json.loads(res["violated_parameters"]) if res.get("violated_parameters") else []
                res["warnings"] = json.loads(res["warnings"]) if res.get("warnings") else []
                return res
            return None

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retrieve currently active alert events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alert_events WHERE status = 'ACTIVE' ORDER BY last_detected DESC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["affected_parameters"] = json.loads(d["affected_parameters"]) if d.get("affected_parameters") else []
                results.append(d)
            return results

    def clear_all_data(self) -> None:
        """Clear database tables (useful for test resets)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM readings")
            cursor.execute("DELETE FROM readiness_assessments")
            cursor.execute("DELETE FROM alert_events")
            conn.commit()
