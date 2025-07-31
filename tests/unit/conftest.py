import pytest
from src.utils import model_utils

model_manager = model_utils.model_manager

@pytest.fixture(scope="session", autouse=True)
def load_model_components():
    """Charge le modèle et le scaler une seule fois avant tous les tests."""
    loaded = model_manager.load_all()
    if not loaded:
        pytest.fail("❌ Impossible de charger le modèle et le scaler.")
    return model_manager
