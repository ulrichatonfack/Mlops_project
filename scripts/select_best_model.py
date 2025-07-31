import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import shutil

# Charger les données
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

def evaluate_model(model_path, scaler_path, X_test, y_test):
    """Évalue un modèle avec son scaler sur les données de test"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Appliquer le scaler aux données de test
    X_test_scaled = scaler.transform(X_test)
    
    # Faire les prédictions
    predictions = model.predict(X_test_scaled)
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions),
        'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
    }
    
    return metrics, model

def select_best_model():
    """Compare tous les modèles et sélectionne le meilleur"""
    models_dir = Path('models/trained')
    scaler_path = Path('data/processed/scaler.pkl')
    
    # Vérifier que le scaler existe
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    results = {}
    models = {}
    
    # Évaluer tous les modèles (sauf best_model.pkl pour éviter la récursion)
    model_files = [f for f in models_dir.glob('*.pkl') if f.name != 'best_model.pkl']
    
    if not model_files:
        raise FileNotFoundError("No model files found in models/trained/")
    
    print(f"Found {len(model_files)} models to evaluate")
    print(f"Using scaler: {scaler_path}")
    print("=" * 50)
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"Évaluation de {model_name}...")
        
        try:
            metrics, model = evaluate_model(model_file, scaler_path, X_test, y_test)
            results[model_name] = metrics
            models[model_name] = model
            
            print(f"  MAE: {metrics['mae']:,.2f}")
            print(f"  RMSE: {metrics['rmse']:,.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Erreur avec {model_name}: {e}")
            continue
    
    if not results:
        raise ValueError("No models could be evaluated successfully")
    
    # Sélectionner le meilleur (critère : R² le plus élevé)
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"R² Score: {best_metrics['r2']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:,.2f}")
    print(f"MAE: {best_metrics['mae']:,.2f}")
    print(f"MAPE: {best_metrics['mape']:.2f}%")
    
    # Sauvegarder le meilleur modèle
    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"✅ Meilleur modèle sauvegardé: {best_model_path}")
    
    # Copier le scaler vers le dossier des modèles pour faciliter le déploiement
    best_scaler_path = models_dir / "best_scaler.pkl"
    shutil.copy2(scaler_path, best_scaler_path)
    print(f"✅ Scaler copié: {best_scaler_path}")

    # Sauvegarder les métadonnées
    metadata = {
        'model_name': best_model_name,
        'model_file': 'best_model.pkl',
        'scaler_file': 'best_scaler.pkl',
        'metrics': best_metrics,
        'feature_names': list(X_test.columns),
        'training_date': str(pd.Timestamp.now()),
        'data_shape': {
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(X_test.columns)
        },
        'all_results': results
    }
    
    metadata_path = models_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Métadonnées sauvegardées: {metadata_path}")
    
    return best_model_name, best_metrics

if __name__ == "__main__":
    try:
        best_name, best_metrics = select_best_model()
        print(f"\n✅ Sélection terminée avec succès!")
        print(f"Meilleur modèle: {best_name} (R² = {best_metrics['r2']:.4f})")
    except Exception as e:
        print(f"❌ Erreur lors de la sélection: {e}")
        raise
    