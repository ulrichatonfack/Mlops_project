import pytest
import numpy as np
from src.utils.model_utils import model_manager, make_prediction, get_feature_names, is_model_ready

@pytest.fixture
def sample_input():
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

def test_feature_names():
    features = get_feature_names()
    assert isinstance(features, list)
    assert "bedrooms" in features

def test_model_loading():
    result = model_manager.load_all()
    assert isinstance(result, bool)

def test_prediction_structure(sample_input):
    if not is_model_ready():
        pytest.skip("⚠️ Modèle non chargé, test ignoré")

    result = make_prediction(sample_input)
    assert "predicted_price" in result
    assert "confidence_interval" in result
    assert "lower" in result["confidence_interval"]
    assert "upper" in result["confidence_interval"]
    assert result["predicted_price"] >= 0
