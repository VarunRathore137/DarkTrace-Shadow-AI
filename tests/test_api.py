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

def test_network_analysis_endpoint():
    response = client.post("/network_analysis", json={
        "user_id": "test_user_1",
        "interactions": [
            {"sender_id": "test_user_1", "receiver_id": "test_user_2", "weight": 5},
            {"sender_id": "test_user_2", "receiver_id": "test_user_3", "weight": 2}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert "metrics" in data
    assert len(data["nodes"]) >= 3
    assert len(data["edges"]) == 2
    assert "pagerank" in data["nodes"][0]
    assert "degree_centrality" in data["nodes"][0]
