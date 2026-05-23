from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_mock():
    response = client.post("/predict", json={"message": "buy weed plug"})
    if response.status_code == 503:
        return # Skip if model not loaded
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_level" in data
    assert "features" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_semantic_search():
    response = client.post("/semantic_search", json={"query": "weed delivery", "limit": 2})
    if response.status_code == 503:
        # ChromaDB might not be available or model not loaded
        return
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    
def test_semantic_search_with_filters():
    response = client.post("/semantic_search", json={
        "query": "cash only",
        "limit": 1,
        "platform": "Telegram",
        "risk_level": "high"
    })
    if response.status_code == 503:
        return
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    if data["results"]:
        assert data["results"][0]["platform"] == "Telegram"
        assert data["results"][0]["risk_level"] == "high"
