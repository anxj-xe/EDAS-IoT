"""Unit and integration tests for FastAPI ingestion endpoints and simulator in EDAS-IoT."""

from fastapi.testclient import TestClient
import pytest
from src.ingestion.api import app
from src.ingestion.simulator import SensorSimulator


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Health check endpoint returns status UP and component metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "timestamp" in data
    assert "baseline_status" in data


def test_receive_valid_telemetry(client):
    """Valid sensor reading returns 201 Created and structured assessment."""
    payload = {
        "device_id": "test_esp32_node",
        "timestamp": "2026-10-13T10:30:00Z",
        "temperature_c": 21.5,
        "humidity_pct": 52.0,
        "pressure_hpa": 1013.2,
        "sequence_number": 1,
    }
    response = client.post("/api/readings", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "PROCESSED"
    assert "reading_id" in data
    assert data["is_valid"] is True
    assert "readiness_status" in data


def test_invalid_http_payload_error(client):
    """Case 14: Malformed HTTP payloads return 422 Unprocessable Entity."""
    # Missing required timestamp and device_id
    bad_payload = {
        "temperature_c": 25.0
    }
    response = client.post("/api/readings", json=bad_payload)
    assert response.status_code == 422


def test_simulator_runs_without_hardware():
    """Case 15: Sensor simulator steps without physical hardware and updates database."""
    simulator = SensorSimulator()
    # Step once
    result = simulator.step()
    assert result is not None
    assert result.raw.device_id is not None
    assert result.raw.timestamp is not None
    assert result.raw.temperature_c is not None

    # Step with on-the-fly injected anomaly
    simulator.inject_anomaly("TEMPERATURE_SPIKE", duration_steps=2)
    step_anom = simulator.step()
    assert step_anom is not None
    assert step_anom.raw.temperature_c > 30.0 # Spiked temperature
