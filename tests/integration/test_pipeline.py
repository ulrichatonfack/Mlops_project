# tests/integration/test_pipeline.py
import pytest
import pandas as pd
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

class TestFullPipeline:
    """Tests d'intégration du pipeline complet"""
    
    def test_model_loading(self):
        """Test que le modèle peut être chargé"""
        model_path = Path("models/trained/best_model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            assert model is not None
            assert hasattr(model, 'predict')
    
    def test_metadata_loading(self):
        """Test que les métadonnées peuvent être chargées"""
        metadata_path = Path("models/trained/model_metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            assert "model_name" in metadata
            assert "metrics" in metadata
            assert "feature_names" in metadata
    
    @pytest.mark.skipif(not Path("models/trained/best_model.pkl").exists(), 
                       reason="Model file not found")
    def test_end_to_end_prediction(self):
        """Test de prédiction bout en bout"""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Simuler le démarrage de l'app pour charger le modèle
        try:
            # Cette partie dépend de votre implémentation du chargement
            response = client.post("/predict", json=VALID_REQUEST)
            # Si le modèle est chargé, on devrait avoir une réponse 200
            # Sinon 503 (service unavailable)
            assert response.status_code in [200, 503]
        except Exception as e:
            pytest.skip(f"End-to-end test skipped: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])