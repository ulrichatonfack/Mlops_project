import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def sample_request():
    return {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 2000,
        "sqft_lot": 8000,
        "floors": 1,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_basement": 0,
        "city_mean_price": 500000,
        "house_age": 25,
        "renovated": 0,
        "age_since_renov": 25
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint(sample_request):
    response = client.post("/predict", json=sample_request)
    assert response.status_code in [200, 503]  # 503 si modèle non chargé
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "confidence_interval" in data
