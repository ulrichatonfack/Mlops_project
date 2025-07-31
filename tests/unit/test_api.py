import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.api.main import app

client = TestClient(app)

# Données de test valides
VALID_REQUEST = {
    "bedrooms": 3.0,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 7500,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_basement": 0,
    "city_mean_price": 450000.0,
    "house_age": 49,
    "renovated": 0,
    "age_since_renov": 49
}

class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_endpoint_model_not_loaded(self):
        """Test health check quand le modèle n'est pas chargé"""
        with patch('src.api.main.model', None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
            assert "timestamp" in data
            assert data["version"] == "1.0.0"
    
    @patch('src.api.main.model', MagicMock())
    def test_health_endpoint_model_loaded(self):
        """Test health check quand le modèle est chargé"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

class TestRootEndpoint:
    """Tests pour l'endpoint racine /"""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "House Price Prediction API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

class TestModelInfoEndpoint:
    """Tests pour l'endpoint /model/info"""
    
    def test_model_info_no_metadata(self):
        """Test model info sans métadonnées"""
        with patch('src.api.main.model_metadata', None):
            response = client.get("/model/info")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    @patch('src.api.main.model_metadata', {
        'model_name': 'ridge_best',
        'metrics': {'r2': 0.072, 'mae': 156387.68},
        'feature_names': ['bedrooms', 'bathrooms'],
        'training_date': '2025-07-30'
    })
    def test_model_info_with_metadata(self):
        """Test model info avec métadonnées"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "ridge_best"
        assert "metrics" in data
        assert "feature_names" in data

class TestPredictionEndpoint:
    """Tests pour l'endpoint /predict"""
    
    def test_predict_model_not_loaded(self):
        """Test prédiction sans modèle chargé"""
        with patch('src.api.main.model', None):
            response = client.post("/predict", json=VALID_REQUEST)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    @patch('src.api.main.model')
    @patch('src.api.main.log_prediction')
    def test_predict_valid_request(self, mock_log, mock_model):
        """Test prédiction avec requête valide"""
        # Mock du modèle
        mock_model.predict.return_value = [450000.0]
        
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_price" in data
        assert "confidence_interval" in data
        assert "model_version" in data
        assert "request_id" in data
        assert data["predicted_price"] == 450000.0
        
        # Vérifier que le logging a été appelé
        mock_log.assert_called_once()
    
    def test_predict_invalid_bedrooms(self):
        """Test avec nombre de chambres invalide"""
        invalid_request = VALID_REQUEST.copy()
        invalid_request["bedrooms"] = -1
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_sqft_living(self):
        """Test avec superficie invalide"""
        invalid_request = VALID_REQUEST.copy()
        invalid_request["sqft_living"] = 50  # Trop petit
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_predict_missing_field(self):
        """Test avec champ manquant"""
        incomplete_request = VALID_REQUEST.copy()
        del incomplete_request["bedrooms"]
        
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422
    
    @patch('src.api.main.model')
    def test_predict_model_exception(self, mock_model):
        """Test gestion d'erreur du modèle"""
        mock_model.predict.side_effect = Exception("Model error")
        
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]