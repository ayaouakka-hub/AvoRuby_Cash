"""
config.py — Centralized Configuration for AvoRuby Cash
Loads all settings from the .env file.
Every module imports from here
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Walk up from this file (src/) to find .env at project root
_this_dir = Path(__file__).resolve().parent                # src/
_project_root = _this_dir.parent.parent                    # AvoRuby_Cash_Project/
_env_path = _project_root / ".env"

load_dotenv(dotenv_path=_env_path)

# OLLAMA 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")

# ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")

# Data Paths
DATA_AGRICOLE_PATH = os.getenv("DATA_AGRICOLE_PATH")
MODELS_DIR = os.getenv("MODELS_DIR")

# Model Filenames
XGBOOST_CREDIT_MODEL = os.getenv("XGBOOST_CREDIT_MODEL")
XGBOOST_PRIX_AVOCAT_MODEL = os.getenv("XGBOOST_PRIX_AVOCAT_MODEL")
XGBOOST_PRIX_FRUITS_MODEL = os.getenv("XGBOOST_PRIX_FRUITS_MODEL")
XGBOOST_PRODUCTION_MODEL = os.getenv("XGBOOST_PRODUCTION_MODEL")
SARIMA_NDWI_MODEL = os.getenv("SARIMA_NDWI_MODEL")
LSTM_VPD_MODEL = os.getenv("LSTM_VPD_MODEL")
SCALER_X_LSTM = os.getenv("SCALER_X_LSTM")
SCALER_Y_LSTM = os.getenv("SCALER_Y_LSTM")

# RAG Settings
RAG_TOP_K = int(os.getenv("RAG_TOP_K"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE"))

# Derived Paths (convenience helpers) 
def model_path(filename: str) -> str:
    """Return the absolute path to a model file inside MODELS_DIR."""
    return os.path.join(MODELS_DIR, filename)
