from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_mock():
    response = client.post("/predict", json={"message": "buy weed plug"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_level" in data
    assert "features" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
