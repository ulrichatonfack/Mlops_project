from model_utils import model_manager

# Charger tous les composants
if model_manager.load_all():
    print("✅ Modèle et scaler chargés")
    
    # Test avec des données factices
    test_data = {
        "bedrooms": 3, "bathrooms": 2, "sqft_living": 2000,
        "sqft_lot": 8000, "floors": 1, "waterfront": 0,
        "view": 0, "condition": 3, "sqft_basement": 0,
        "city_mean_price": 500000, "house_age": 25,
        "renovated": 0, "age_since_renov": 25
    }
    
    result = model_manager.predict_with_confidence(test_data)
    print(f"Prédiction: ${result['prediction']:,.2f}")
else:
    print("❌ Échec du chargement")