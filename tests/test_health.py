from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Test that the health endpoint returns a 200 status code."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_ok_status():
    """Test that the health endpoint returns status ok."""
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
