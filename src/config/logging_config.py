import structlog
import logging
import json
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Configure le logging structuré pour l'API"""
    
    # Créer le dossier de logs s'il n'existe pas
    Path("logs/api_logs").mkdir(parents=True, exist_ok=True)
    
    # Configuration du logger standard Python
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("logs/api_logs/api.log"),
            logging.StreamHandler()
        ]
    )
    
    # Configuration de structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()

def log_prediction(logger, request_data, prediction, duration, request_id):
    """Log une prédiction avec le format requis"""
    logger.info(
        "prediction_made",
        timestamp=datetime.now().isoformat(),
        request_id=request_id,
        features=request_data,
        prediction=float(prediction),
        duration_ms=duration * 1000,  # Convertir en millisecondes
        model_version="v1.0"
    )