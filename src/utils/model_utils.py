"""
Utilitaires pour la gestion des modèles de machine learning
"""
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestionnaire des modèles et du preprocessing"""
    
    def __init__(self, models_dir: str = "models/trained", data_dir: str = "data/processed"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_order = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "sqft_basement", "city_mean_price",
            "house_age", "renovated", "age_since_renov"
        ]
    
    def load_model(self, model_name: str = "best_model.pkl") -> bool:
        """
        Charge le modèle spécifié
        
        Args:
            model_name: Nom du fichier du modèle à charger
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def load_scaler(self, scaler_name: str = "scaler.pkl") -> bool:
        """
        Charge le scaler
        
        Args:
            scaler_name: Nom du fichier du scaler à charger
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            scaler_path = self.data_dir / scaler_name
            
            if not scaler_path.exists():
                logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded successfully: {scaler_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scaler {scaler_name}: {str(e)}")
            return False
    
    def load_metadata(self, metadata_name: str = "model_metadata.json") -> bool:
        """
        Charge les métadonnées du modèle
        
        Args:
            metadata_name: Nom du fichier de métadonnées
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            metadata_path = self.models_dir / metadata_name
            
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info("Metadata loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return False
    
    def load_all(self) -> bool:
        """
        Charge le modèle, le scaler et les métadonnées
        
        Returns:
            bool: True si tous les chargements ont réussi
        """
        model_loaded = self.load_model()
        scaler_loaded = self.load_scaler()
        metadata_loaded = self.load_metadata()  # Optionnel
        
        if model_loaded and scaler_loaded:
            logger.info("All components loaded successfully")
            return True
        else:
            logger.error("Failed to load some components")
            return False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Préprocesse les données d'entrée
        
        Args:
            input_data: Dictionnaire avec les features d'entrée
            
        Returns:
            np.ndarray: Données préprocessées prêtes pour la prédiction
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Call load_scaler() first.")
        
        # Créer DataFrame avec l'ordre correct des features
        df_input = pd.DataFrame([input_data])[self.feature_order]
        
        # Appliquer le scaler
        scaled_data = self.scaler.transform(df_input)
        
        return scaled_data
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Fait une prédiction sur les données d'entrée
        
        Args:
            input_data: Dictionnaire avec les features d'entrée
            
        Returns:
            float: Prix prédit
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Préprocesser les données
        processed_data = self.preprocess_input(input_data)
        
        # Faire la prédiction
        prediction = self.model.predict(processed_data)[0]
        
        # S'assurer que la prédiction est positive
        return max(0, float(prediction))
    
    def predict_with_confidence(self, input_data: Dict[str, Any], 
                              confidence_level: float = 0.15) -> Dict[str, Any]:
        """
        Fait une prédiction avec intervalle de confiance
        
        Args:
            input_data: Dictionnaire avec les features d'entrée
            confidence_level: Niveau de confiance (défaut: 15%)
            
        Returns:
            Dict contenant la prédiction et l'intervalle de confiance
        """
        prediction = self.predict(input_data)
        
        # Calcul simple de l'intervalle de confiance
        margin = prediction * confidence_level
        
        return {
            "prediction": prediction,
            "confidence_interval": {
                "lower": max(0, prediction - margin),
                "upper": prediction + margin
            }
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Valide les données d'entrée
        
        Args:
            input_data: Dictionnaire avec les features d'entrée
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Vérifier que toutes les features requises sont présentes
        missing_features = set(self.feature_order) - set(input_data.keys())
        if missing_features:
            return False, f"Missing features: {list(missing_features)}"
        
        # Vérifier les types et valeurs
        try:
            for feature, value in input_data.items():
                if feature in self.feature_order:
                    # Convertir en float et vérifier
                    float_val = float(value)
                    
                    # Vérifications spécifiques
                    if feature in ['bedrooms', 'bathrooms'] and float_val < 0:
                        return False, f"{feature} cannot be negative"
                    
                    if feature in ['sqft_living', 'sqft_lot'] and float_val <= 0:
                        return False, f"{feature} must be positive"
                    
                    if feature in ['waterfront', 'renovated'] and float_val not in [0, 1]:
                        return False, f"{feature} must be 0 or 1"
                    
                    if feature == 'condition' and not (1 <= float_val <= 5):
                        return False, f"{feature} must be between 1 and 5"
                    
                    if feature == 'view' and not (0 <= float_val <= 4):
                        return False, f"{feature} must be between 0 and 4"
            
            return True, None
            
        except (ValueError, TypeError) as e:
            return False, f"Invalid data type: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle chargé
        
        Returns:
            Dict avec les informations du modèle
        """
        info = {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "metadata_loaded": self.metadata is not None,
            "feature_order": self.feature_order
        }
        
        if self.metadata:
            info.update({
                "model_name": self.metadata.get('model_name'),
                "metrics": self.metadata.get('metrics'),
                "training_date": self.metadata.get('training_date')
            })
        
        return info


# Instance globale pour l'API
model_manager = ModelManager()


def load_model_components() -> bool:
    """
    Fonction utilitaire pour charger tous les composants du modèle
    
    Returns:
        bool: True si le chargement a réussi
    """
    return model_manager.load_all()


def make_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction utilitaire pour faire une prédiction
    
    Args:
        input_data: Dictionnaire avec les features d'entrée
        
    Returns:
        Dict avec la prédiction et les métadonnées
    """
    # Valider les données
    is_valid, error_msg = model_manager.validate_input(input_data)
    if not is_valid:
        raise ValueError(f"Invalid input data: {error_msg}")
    
    # Faire la prédiction avec intervalle de confiance
    result = model_manager.predict_with_confidence(input_data)
    
    return {
        "predicted_price": round(result["prediction"], 2),
        "confidence_interval": {
            "lower": round(result["confidence_interval"]["lower"], 2),
            "upper": round(result["confidence_interval"]["upper"], 2)
        }
    }


def get_feature_names() -> list:
    """
    Retourne la liste des noms des features dans l'ordre attendu
    
    Returns:
        list: Liste des noms des features
    """
    return model_manager.feature_order.copy()


def is_model_ready() -> bool:
    """
    Vérifie si le modèle est prêt pour les prédictions
    
    Returns:
        bool: True si le modèle et le scaler sont chargés
    """
    return model_manager.model is not None and model_manager.scaler is not None