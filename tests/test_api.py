"""Tests for SETU API endpoints."""

from fastapi.testclient import TestClient
from setu.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns HTTP 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_triage_queue_endpoint():
    """Verify GET /api/triage returns a list."""
    response = client.get("/api/triage")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_bridges_endpoint():
    """Verify GET /api/bridges returns a list."""
    response = client.get("/api/bridges")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
