import time
import uuid
from datetime import datetime
from pathlib import Path
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ajouter le chemin racine pour les imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.model_utils import model_manager, make_prediction, get_feature_names, is_model_ready
from src.api.schemas.request_schemas import HousePredictionRequest
from src.api.schemas.response_schemas import PredictionResponse, HealthResponse
from src.config.logging_config import setup_logging, log_prediction

# Initialisation du logger
logger = setup_logging()

# ✅ Nouveau gestionnaire lifespan (remplace on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model components...")
        if model_manager.load_all():
            logger.info("✅ Model components loaded successfully")
        else:
            logger.error("❌ Failed to load model components")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e
    yield  # Obligatoire (sinon FastAPI ne démarre pas)

# Initialisation FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API pour prédire le prix des maisons avec modèle de machine learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour logger toutes les requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)

    duration = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "request_processed",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration_ms=duration
    )

    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_ready = is_model_ready()
    return HealthResponse(
        status="healthy" if model_ready else "unhealthy",
        model_loaded=model_ready,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/model/info")
async def model_info():
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    info = model_manager.get_model_info()
    return {
        "model_name": info.get('model_name', 'unknown'),
        "metrics": info.get('metrics'),
        "feature_names": get_feature_names(),
        "training_date": info.get('training_date'),
        "model_loaded": info['model_loaded'],
        "scaler_loaded": info['scaler_loaded'],
        "version": "1.0.0"
    }

@app.get("/model/features")
async def get_model_features():
    return {
        "features": get_feature_names(),
        "total_features": len(get_feature_names()),
        "description": "Features required for house price prediction"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: HousePredictionRequest, http_request: Request):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])

    try:
        input_data = request.dict()

        prediction_result = make_prediction(input_data)
        duration = time.time() - start_time

        log_prediction(
            logger=logger,
            request_data=input_data,
            prediction=prediction_result["predicted_price"],
            duration=duration,
            request_id=request_id
        )

        return PredictionResponse(
            predicted_price=prediction_result["predicted_price"],
            confidence_interval=prediction_result["confidence_interval"],
            model_version="v1.0",
            request_id=request_id
        )

    except ValueError as ve:
        logger.warning("Validation error", request_id=request_id, error=str(ve))
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")

    except Exception as e:
        logger.error("Prediction failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: list[HousePredictionRequest], http_request: Request):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(requests) > 100:
        raise HTTPException(status_code=413, detail="Too many requests (max 100)")

    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    start_time = time.time()

    results = []
    for i, req in enumerate(requests):
        try:
            prediction_result = make_prediction(req.dict())
            results.append({
                "index": i,
                "predicted_price": prediction_result["predicted_price"],
                "confidence_interval": prediction_result["confidence_interval"],
                "status": "success"
            })
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e),
                "status": "error"
            })

    logger.info("batch_prediction_completed", request_id=request_id)

    return {
        "results": results,
        "total_requests": len(requests),
        "successful_predictions": sum(1 for r in results if r["status"] == "success"),
        "request_id": request_id
    }

@app.get("/")
async def root():
    model_info = model_manager.get_model_info() if is_model_ready() else {}
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "model": model_info.get('model_name', 'Not loaded'),
        "model_ready": is_model_ready(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "model_features": "/model/features",
            "docs": "/docs"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error("Unhandled exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
