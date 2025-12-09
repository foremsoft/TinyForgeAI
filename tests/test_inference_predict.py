from fastapi.testclient import TestClient

from inference_server.app import app

client = TestClient(app)


def test_predict_returns_200():
    """Test that POST /predict returns HTTP 200."""
    response = client.post("/predict", json={"input": "hello"})
    assert response.status_code == 200


def test_predict_returns_correct_keys():
    """Test that POST /predict returns output and confidence keys."""
    response = client.post("/predict", json={"input": "hello"})
    data = response.json()
    assert "output" in data
    assert "confidence" in data


def test_predict_reverses_input():
    """Test that the stub model reverses the input string."""
    response = client.post("/predict", json={"input": "hello"})
    data = response.json()
    assert data["output"] == "olleh"


def test_predict_confidence_in_valid_range():
    """Test that confidence score is between 0 and 1."""
    response = client.post("/predict", json={"input": "hello"})
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_health_returns_200():
    """Test that GET /health returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_correct_body():
    """Test that GET /health returns the expected JSON body."""
    response = client.get("/health")
    assert response.json() == {"status": "ok", "service": "inference-server"}
