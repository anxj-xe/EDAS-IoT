"""FastAPI hardware ingestion and telemetry service for EDAS-IoT.

Receives sensor readings via HTTP POST from ESP32 nodes and simulators,
executes the unified processing pipeline, and persists all states in SQLite.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.common.config import get_config
from src.common.schemas import (
    AlertEvent,
    ProcessedReading,
    RawSensorReading,
    ReadinessAssessment,
    ReadinessStatus,
    ValidationStatus,
)
from src.detection.baseline import BaselineCalibrator, BaselineModel
from src.detection.isolation_forest import IsolationForestDetector
from src.ingestion.simulator import SensorSimulator
from src.processing.pipeline import EDASPipeline
from src.storage.repository import DatabaseRepository


app = FastAPI(
    title="EDAS-IoT Telemetry & Ingestion API",
    description="Environmental Anomaly Detection and IoT Deployment Readiness System API",
    version="1.0.0",
)

# Enable CORS for local dashboards or web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline, repository, and simulator instances
config = get_config()
repo = DatabaseRepository()

models_dir = config.root_dir / "models"
baseline_file = models_dir / "baseline.json"
model_file = models_dir / "isolation_forest.joblib"

baseline = BaselineModel()
if baseline_file.exists():
    calibrator = BaselineCalibrator()
    baseline = calibrator.load_baseline(baseline_file)

detector = IsolationForestDetector()
if model_file.exists():
    detector.load_model(model_file)

pipeline = EDASPipeline(baseline=baseline, ml_detector=detector)
simulator = SensorSimulator(pipeline=pipeline, repository=repo)


# Response Schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    baseline_status: str
    ml_model_fitted: bool
    active_profile: str


class ReadingIngestResponse(BaseModel):
    status: str
    reading_id: int
    is_valid: bool
    is_anomaly: bool
    severity: str
    explanation: str
    readiness_status: str


class AnomalyInjectionRequest(BaseModel):
    anomaly_type: str = Field(..., description="e.g. TEMPERATURE_SPIKE, STUCK_SENSOR, MULTIVARIATE_ANOMALY")
    duration_steps: int = Field(5, description="Number of stream steps to maintain injection")


@app.get("/health", response_model=HealthResponse)
def get_health():
    """Application health and component status check."""
    return HealthResponse(
        status="UP",
        timestamp=datetime.now(timezone.utc).isoformat(),
        baseline_status=pipeline.baseline.status,
        ml_model_fitted=pipeline.ml_detector.is_fitted,
        active_profile=pipeline.readiness_assessor.device_profile.name if pipeline.readiness_assessor.device_profile else "None",
    )


@app.post("/api/readings", response_model=ReadingIngestResponse, status_code=status.HTTP_201_CREATED)
def receive_sensor_reading(payload: RawSensorReading):
    """Receive a timestamped sensor reading from ESP32 or external sensor."""
    try:
        processed, readiness, alert = pipeline.process_reading(payload)

        # Store in SQLite database
        reading_id = repo.insert_processed_reading(processed)
        repo.insert_readiness_assessment(readiness)
        if alert:
            repo.upsert_alert_event(alert)

        return ReadingIngestResponse(
            status="PROCESSED",
            reading_id=reading_id,
            is_valid=processed.validation.is_valid,
            is_anomaly=processed.is_overall_anomaly,
            severity=processed.severity.value,
            explanation=processed.explanation,
            readiness_status=readiness.status.value,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing sensor telemetry: {str(e)}",
        )


@app.get("/api/readings/latest")
def get_latest_reading(device_id: Optional[str] = Query(None)):
    """Retrieve the most recent processed sensor reading."""
    reading = repo.get_latest_reading(device_id)
    if not reading:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No readings found in database.")
    return reading


@app.get("/api/readings/recent")
def get_recent_readings(limit: int = Query(50, ge=1, le=500), device_id: Optional[str] = Query(None)):
    """Retrieve recent readings for time-series visualization."""
    return repo.get_recent_readings(limit=limit, device_id=device_id)


@app.get("/api/anomalies")
def get_anomalies(limit: int = Query(50, ge=1, le=500)):
    """Retrieve list of flagged anomalous readings."""
    return repo.get_anomalies(limit=limit)


@app.get("/api/readiness")
def get_readiness():
    """Retrieve the latest IoT deployment readiness assessment."""
    readiness = repo.get_latest_readiness()
    if not readiness:
        # Fallback to in-memory readiness assessment
        curr = pipeline.readiness_assessor.assess_window("default", [], pipeline.readiness_assessor.device_profile)
        return curr.model_dump()
    return readiness


@app.get("/api/alerts")
def get_alerts():
    """Retrieve currently active alert events."""
    return repo.get_active_alerts()


# Simulation Control Endpoints
@app.post("/api/simulate/step")
def simulate_step():
    """Advance the streaming simulator by one step."""
    processed = simulator.step()
    if not processed:
        return {"status": "NO_DATA"}
    return {
        "status": "STEPPED",
        "timestamp": processed.raw.timestamp,
        "temperature_c": processed.raw.temperature_c,
        "humidity_pct": processed.raw.humidity_pct,
        "pressure_hpa": processed.raw.pressure_hpa,
        "is_anomaly": processed.is_overall_anomaly,
        "severity": processed.severity.value,
        "explanation": processed.explanation,
    }


@app.post("/api/simulate/inject")
def inject_anomaly(req: AnomalyInjectionRequest):
    """Inject a controlled anomaly into upcoming simulation steps."""
    simulator.inject_anomaly(req.anomaly_type, duration_steps=req.duration_steps)
    return {
        "status": "ANOMALY_SCHEDULED",
        "anomaly_type": req.anomaly_type,
        "duration_steps": req.duration_steps,
    }


@app.get("/api/profiles")
def get_device_profiles():
    """Return all available device deployment profiles."""
    profiles = config.get_device_profiles()
    return {k: p.model_dump() for k, p in profiles.items()}
