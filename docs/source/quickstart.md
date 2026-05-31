# Quick Start

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.14 or above |
| Ollama | Latest |
| LLM model | `gemma:4b` |
| Embedding model | `nomic-embed-text:latest` |
| RAM | 8 GB minimum recommended |

## Installation Steps

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ayaouakka-hub/AvoRuby_Cash.git
cd AvoRuby_Cash
```

### Step 2 — Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Ollama Models

First, ensure Ollama is installed and running on your system, then:

```bash
ollama pull gemma:4b
ollama pull nomic-embed-text:latest
```

### Step 5 — Configure Environment Variables

Create a `.env` file at the project root:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest

CHROMA_DB_PATH=./AvoRuby_Back/chroma_db
CHROMA_COLLECTION_NAME=avoruby_agricole

DATA_AGRICOLE_PATH=./AvoRuby_Back/data/data_agricole
MODELS_DIR=./AvoRuby_Back/models

XGBOOST_CREDIT_MODEL=xgboost_credit_v6.pkl
XGBOOST_PRIX_AVOCAT_MODEL=xgboost_prix_avocat.pkl
XGBOOST_PRIX_FRUITS_MODEL=xgboost_prix_fruits_rouges.pkl
XGBOOST_PRODUCTION_MODEL=xgboost_production.pkl
SARIMA_NDWI_MODEL=sarima_ndwi.pkl
LSTM_VPD_MODEL=lstm_vpd.pt
```

### Step 6 — Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` by default.

## Troubleshooting

**Issue:** `ollama: command not found`
- **Solution:** Install Ollama from https://ollama.ai and restart your terminal.

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`
- **Solution:** Ensure you activated the virtual environment and ran `pip install -r requirements.txt`.

**Issue:** `ChromaDB connection error`
- **Solution:** Verify `CHROMA_DB_PATH` exists and is writable.

**Issue:** `Ollama model not found`
- **Solution:** Run `ollama pull gemma:4b` and `ollama pull nomic-embed-text:latest` again.
