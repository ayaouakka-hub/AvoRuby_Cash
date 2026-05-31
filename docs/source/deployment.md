# Deployment & Application

## Streamlit Dashboard

The application provides an interactive web interface for credit evaluation built with Streamlit 1.58+.

### Interface Sections

1. **Farmer Profile Form** — Culture, location, soil type, experience, farm size, credit amount, equipment checklist

2. **Credit Decision Banner** — SABC score, decision (APPROUVE / CONDITIONNEL / REFUSE), adjustment annotation

3. **Production Estimates** — Yield (kg/ha), price (MAD/kg), production volume, gross revenue, net revenue, operating charges, insurance coverage

4. **Financing Analysis** — Credit amount, annuity (10-year, 5%), DCR, RCR with color-coded indicators

5. **Model Probabilities** — Bar chart for decision class probabilities

6. **Climate Indicators** — VPD gauge, NDWI gauge, thermal stress days gauge

7. **Equipment Assessment** — Tags for missing bancability-enhancing equipment

8. **Strategic AI Advisor** — On-demand RAG-powered personalized agricultural recommendation

### Design System

- **Color Palette:** Forest green `#0c1f13` / Crimson red `#c0392b` / Off-white `#f4f1ec`
- **Typography:** Playfair Display (headings) + Inter (body)
- **Philosophy:** Typography-first, no decorative icons

## Credit Decision Logic

### 1. Model Decision (XGBoost v6)

The classifier produces a 3-class probability vector. The argmax class is the raw model decision.

### 2. Financing Feasibility Adjustment

When a credit amount (MAD) is provided, the system computes:

**Annuity estimate** (10-year loan, 5% rate):

$$\text{annuity} = \text{credit\_amount} \times 0.1295$$

**Debt Coverage Ratio (DCR):**

$$\text{DCR} = \frac{\text{net\_revenue}}{\text{annuity}}$$

**Revenue-to-Credit Ratio (RCR):**

$$\text{RCR} = \frac{\text{gross\_revenue}}{\text{credit\_amount}}$$

**Decision adjustment rules:**

| Condition | Action |
|---|---|
| DCR ≥ 1.3 AND RCR ≥ 1.0 | Model decision retained |
| 0.8 ≤ DCR < 1.3 OR 0.5 ≤ RCR < 1.0 | APPROUVE downgraded to CONDITIONNEL |
| DCR < 0.8 OR RCR < 0.5 | Decision forced to REFUSE |

## RAG Advisory Engine

**Module:** `AvoRuby_Back/src/assistat/`

The advisory component is a Retrieval-Augmented Generation (RAG) pipeline built on LangChain, ChromaDB, and a locally hosted Ollama LLM.

### Pipeline Architecture

```
Knowledge Base (TXT files)
        |
        | LangChain TextSplitter (chunk=1000, overlap=200)
        v
ChromaDB (nomic-embed-text embeddings, 768-dim)
        |
        | Semantic retrieval (top-k = 3)
        v
Ollama LLM (gemma:4b)
        |
        | Farmer profile + context + structured prompt
        v
Personalized agronomic + financial recommendation
```

### Knowledge Base Coverage

| File | Domain |
|---|---|
| `01_culture_avocat_maroc.txt` | Avocado cultivation, varieties, yields, diseases, prices |
| `02_culture_fruits_rouges_maroc.txt` | Red fruits — full technical guide |
| `03_credit_agricole_maroc_cam.txt` | CAM credit products, SABC, FDA subsidies, MAMDA |
| `04_risque_climatique_irrigation.txt` | VPD, NDWI, water resources, drip irrigation, IoT |
| `05_analyse_financiere_rentabilite.txt` | Cost structures, profitability, business plan templates |
| `06_profils_regionaux_maroc.txt` | Regional agro-climatic profiles (Souss, Gharb, Haouz...) |
| `07_bonnes_pratiques_techniques.txt` | Best practices: bancability, irrigation scheduling, GAP |

### RAG Features

- **Semantic retrieval** based on farmer context and profile
- **Multi-turn context** — answers adapt to previous interactions
- **Fallback handling** — graceful degradation if LLM unavailable
- **Source attribution** — recommendations cite relevant knowledge base sections

## Running the Application

### Local Development

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Production Deployment

```bash
# Using Streamlit Cloud
git push origin master:interface
# (Streamlit automatically detects app.py and deploys)

# Or Docker deployment
docker build -t avoruby-cash .
docker run -p 8501:8501 avoruby-cash
```

### Environment Configuration

Ensure your `.env` file includes:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
CHROMA_DB_PATH=./AvoRuby_Back/chroma_db
DATA_AGRICOLE_PATH=./AvoRuby_Back/data/data_agricole
MODELS_DIR=./AvoRuby_Back/models
```

## API Endpoints

**Inference function:** `AvoRuby_Back.src.models.traiter_dossier_agriculteur()`

Input: Farmer profile dictionary  
Output: Credit decision + SABC score + predictions + probabilities
