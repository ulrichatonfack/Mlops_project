from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionResponse(BaseModel):
    """Schéma pour les réponses de prédiction"""
    
    predicted_price: float = Field(..., description="Prix prédit en USD")
    confidence_interval: Dict[str, float] = Field(..., description="Intervalle de confiance")
    model_version: str = Field(..., description="Version du modèle utilisé")
    request_id: str = Field(..., description="ID unique de la requête")


    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_price": 485000.0,
                "confidence_interval": {"lower": 450000.0, "upper": 520000.0},
                "model_version": "v1.0",
                "request_id": "req_123456789"
            }
        },
        "protected_namespaces": ()
    }

class HealthResponse(BaseModel):
    """Schéma pour la vérification de santé"""
    
    status: str = Field(..., description="État du service")
    model_loaded: bool = Field(..., description="Modèle chargé avec succès")
    timestamp: str = Field(..., description="Timestamp de la vérification")
    version: str = Field(..., description="Version de l'API")
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2025-07-31T12:34:56",
                "version": "1.0.0"
            }
        },
        "protected_namespaces": ()
    }