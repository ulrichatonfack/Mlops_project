from pydantic import BaseModel, Field
from typing import Optional

class HousePredictionRequest(BaseModel):
    """Schéma pour les requêtes de prédiction basé sur vos features exactes"""
    
    bedrooms: float = Field(..., ge=0, le=20, description="Nombre de chambres")
    bathrooms: float = Field(..., ge=0, le=10, description="Nombre de salles de bain")
    sqft_living: int = Field(..., ge=300, le=15000, description="Surface habitable en pieds carrés")
    sqft_lot: int = Field(..., ge=500, le=200000, description="Surface du terrain")
    floors: float = Field(..., ge=1, le=4, description="Nombre d'étages")
    waterfront: int = Field(..., ge=0, le=1, description="Vue sur l'eau (0/1)")
    view: int = Field(..., ge=0, le=4, description="Qualité de la vue (0-4)")
    condition: int = Field(..., ge=1, le=5, description="État de la maison (1-5)")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Surface du sous-sol")
    city_mean_price: float = Field(..., ge=100000, le=2000000, description="Prix moyen de la ville")
    house_age: int = Field(..., ge=0, le=124, description="Âge de la maison")
    renovated: int = Field(..., ge=0, le=1, description="Maison rénovée (0/1)")
    age_since_renov: int = Field(..., ge=0, le=124, description="Années depuis rénovation")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "bedrooms": 3.0,
                "bathrooms": 2.0,
                "sqft_living": 1800,
                "sqft_lot": 7500,
                "floors": 5.0,
                "waterfront": 1,
                "view": 0,
                "condition": 3,
                "sqft_basement": 0,
                "city_mean_price": 450000.0,
                "house_age": 49,
                "renovated": 0,
                "age_since_renov": 49
            }
        }
    }