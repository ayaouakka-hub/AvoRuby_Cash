"""
models.py — Prediction Pipeline for AvoRuby Cash V2
Integrates LSTM, SARIMA, XGBoost (Prix/Prod), XGBoost V6 (Decision).
All config is loaded from config.py.

Model formats (from training notebooks):
  - xgboost_credit_v6.pkl   : dict {model, ordinal_encoder, label_encoder, feature_names, ...}
  - xgboost_production.pkl  : XGBRegressor (25 features)
  - xgboost_prix_avocat.pkl : dict {model, features}
  - xgboost_prix_fruits_rouges.pkl : dict {model, features}
  - sarima_ndwi.pkl          : SARIMAXResultsWrapper
  - lstm_vpd.pt              : OrderedDict (PyTorch state_dict)
"""

import sys
import os
import math
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Ensure project root is on sys.path
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AvoRuby_Back.src.config import (
    MODELS_DIR,
    XGBOOST_CREDIT_MODEL,
    XGBOOST_PRIX_AVOCAT_MODEL,
    XGBOOST_PRIX_FRUITS_MODEL,
    XGBOOST_PRODUCTION_MODEL,
    SARIMA_NDWI_MODEL,
    LSTM_VPD_MODEL,
    SCALER_X_LSTM,
    SCALER_Y_LSTM,
    model_path,
)

logger = logging.getLogger("AvoRubyModels")

#  LSTM Model Definition (matches notebook cell 18)
try:
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        """LSTM for VPD prediction — architecture from training notebook."""
        def __init__(self, input_size, h1=64, h2=32):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, h1, batch_first=True)
            self.drop1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
            self.drop2 = nn.Dropout(0.2)
            self.fc = nn.Linear(h2, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            out = self.drop2(out[:, -1, :])
            return self.fc(out).squeeze()

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — LSTM model will not be loaded.")


#  1. MODEL LOADING
# 13 base features + one-hot(Culture=2, Localisation=7, Type_Sol=3) = 25
PROD_BASE_FEATURES = [
    "Superficie_ha", "Annees_Experience", "Gamma",
    "Irrigation", "Solaire", "IoT_sondes", "Filets_Serres",
    "VPD_moyen_saison", "Jours_stress_thermique",
    "Pluie_saison_mm", "NDWI_moyen", "Surface_barrage_norm",
    "Surface_barrage_specifique_norm",
]

PROD_ONEHOT_COLS = [
    # Culture (sorted alphabetically by pd.get_dummies)
    "Culture_Avocat", "Culture_Fruits_Rouges",
    # Localisation (sorted)
    "Localisation_Agadir", "Localisation_Kenitra", "Localisation_Larache",
    "Localisation_Marrakech", "Localisation_Moulay Bousselham",
    "Localisation_Souss", "Localisation_Tadla",
    # Type_Sol (sorted)
    "Type_Sol_Argileux", "Type_Sol_Limoneux", "Type_Sol_Sableux",
]

PROD_ALL_FEATURES = PROD_BASE_FEATURES + PROD_ONEHOT_COLS  # 25 total

# -- Localisation GPS mapping for Surface_barrage_specifique_norm --
_LOC_BARRAGE_NORM = {
    "Agadir":            0.30,
    "Souss":             0.30,
    "Kenitra":           0.55,
    "Larache":           0.60,
    "Marrakech":         0.40,
    "Moulay Bousselham": 0.58,
    "Tadla":             0.45,
}


def _load_safe(filename):
    """Safely load a joblib-serialized file from MODELS_DIR."""
    if filename is None:
        return None
    p = model_path(filename)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    else:
        logger.warning(f"File not found: {p}")
    return None


def load_all_models():
    """Load all ML models into memory. Returns a dict of model objects."""
    logger.info("Loading models into memory...")

    # -- Credit V6 (dict-wrapped) --
    v6_pkg = _load_safe(XGBOOST_CREDIT_MODEL)
    # -- Production (plain XGBRegressor) --
    xgb_prod = _load_safe(XGBOOST_PRODUCTION_MODEL)
    # -- Prix models (dict-wrapped) --
    prix_avo_pkg = _load_safe(XGBOOST_PRIX_AVOCAT_MODEL)
    prix_fruits_pkg = _load_safe(XGBOOST_PRIX_FRUITS_MODEL)
    # -- SARIMA for NDWI --
    sarima_ndwi = _load_safe(SARIMA_NDWI_MODEL)

    # -- LSTM for VPD --
    lstm_vpd = None
    scaler_x = None
    scaler_y = None
    if TORCH_AVAILABLE and LSTM_VPD_MODEL:
        lstm_path = model_path(LSTM_VPD_MODEL)
        if os.path.exists(lstm_path):
            try:
                state_dict = torch.load(lstm_path, map_location="cpu", weights_only=False)
                # Infer input_size from first LSTM layer weight shape
                input_size = state_dict["lstm1.weight_ih_l0"].shape[1]
                lstm_vpd = LSTMModel(input_size=input_size)
                lstm_vpd.load_state_dict(state_dict)
                lstm_vpd.eval()
                logger.info(f"LSTM VPD loaded (input_size={input_size}).")
                # Load scalers
                scaler_x = _load_safe(SCALER_X_LSTM)
                scaler_y = _load_safe(SCALER_Y_LSTM)
            except Exception as e:
                logger.error(f"Error loading LSTM: {e}")

    models = {
        "v6_pkg":         v6_pkg,
        "xgb_prod":       xgb_prod,
        "prix_avo_pkg":   prix_avo_pkg,
        "prix_fruits_pkg": prix_fruits_pkg,
        "sarima_ndwi":    sarima_ndwi,
        "lstm_vpd":       lstm_vpd,
        "scaler_x_lstm":  scaler_x,
        "scaler_y_lstm":  scaler_y,
    }

    loaded = sum(1 for v in models.values() if v is not None)
    logger.info(f"{loaded}/{len(models)} components loaded.")
    return models


# Global model registry (populated once on import)
_models = load_all_models()

# ══════════════════════════════════════════════════════
# 2. CLIMATE FORECASTS (Dynamisés par Localisation)
# ══════════════════════════════════════════════════════
def _forecast_ndwi(localisation: str) -> float:
    """Forecast NDWI using SARIMA model, dynamically adjusted by region."""
    base_val = 0.2  # default fallback
    sarima = _models.get("sarima_ndwi")
    
    if sarima is not None:
        try:
            forecast = sarima.forecast(steps=1)
            base_val = float(forecast[0])
            logger.info(f"SARIMA NDWI base forecast: {base_val:.4f}")
        except Exception as e:
            logger.warning(f"SARIMA forecast failed: {e}")
            
    # Ajustement géographique pour dynamiser l'interface Streamlit
    offsets = {
        "Agadir": -0.15,
        "Souss": -0.18,
        "Marrakech": -0.20,
        "Tadla": -0.10,
        "Kenitra": +0.05,
        "Larache": +0.08,
        "Moulay Bousselham": +0.06,
    }
    final_val = base_val + offsets.get(localisation, 0.0)
    return round(final_val, 3)


def _forecast_vpd(localisation: str) -> float:
    """Forecast VPD using LSTM model, dynamically adjusted by region."""
    base_val = 2.1  # default fallback
    lstm = _models.get("lstm_vpd")
    scaler_x = _models.get("scaler_x_lstm")
    scaler_y = _models.get("scaler_y_lstm")
    
    if lstm is not None and scaler_x is not None and scaler_y is not None and TORCH_AVAILABLE:
        try:
            n_features = scaler_x.n_features_in_
            dummy_input = np.zeros((1, 1, n_features))
            tensor_in = torch.tensor(dummy_input, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = lstm(tensor_in).numpy().reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_scaled)
            base_val = float(pred[0, 0])
            logger.info(f"LSTM VPD base forecast: {base_val:.4f}")
        except Exception as e:
            logger.warning(f"LSTM VPD forecast failed: {e}")
            
    # Ajustement géographique (Souss/Marrakech = plus de stress thermique)
    offsets = {
        "Agadir": +0.4,
        "Souss": +0.6,
        "Marrakech": +0.7,
        "Tadla": +0.3,
        "Kenitra": -0.3,
        "Larache": -0.4,
        "Moulay Bousselham": -0.3,
    }
    final_val = base_val + offsets.get(localisation, 0.0)
    return round(final_val, 3)


#  3. PRIX PREDICTION (time-series models)
def _predict_prix(culture: str, localisation: str = "Souss") -> float:
    import datetime
    now = datetime.datetime.now()
    mois = now.month
    annee = now.year
    sin_mois = math.sin(2 * math.pi * mois / 12)
    cos_mois = math.cos(2 * math.pi * mois / 12)
    eur_mad = 10.85

    # Ajustement régional réaliste (proximité export, transport)
    regional_premium = {
        "Agadir":            +1.5,
        "Souss":             +1.2,
        "Marrakech":         +0.5,
        "Tadla":             +0.3,
        "Kenitra":           +2.0,  # proche port Casablanca
        "Larache":           +1.8,
        "Moulay Bousselham": +1.6,
    }

    if culture.lower() == "avocat":
        pkg = _models.get("prix_avo_pkg")
        if pkg is None:
            return 20.0 + regional_premium.get(localisation, 0)
        saison_export = 1 if mois in [10, 11, 12, 1, 2, 3] else 0
        # Lags dynamiques basés sur le mois (simulation courbe saisonnière)
        base_lag = 15.0 + 3.0 * cos_mois  # varie entre 12 et 18
        features = np.array([[
            mois, annee, sin_mois, cos_mois, eur_mad, saison_export,
            0.03 + 0.02 * abs(sin_mois),  # volatilite saisonnière
            base_lag,
            base_lag - 0.5,
            base_lag - 1.0,
            base_lag - 0.3,
            0.01 + 0.01 * saison_export,
        ]])
        try:
            prix_base = float(pkg["model"].predict(features)[0])
            return round(prix_base + regional_premium.get(localisation, 0), 2)
        except Exception as e:
            logger.warning(f"Prix avocat prediction failed: {e}")
            return round(20.0 + regional_premium.get(localisation, 0), 2)

    else:
        pkg = _models.get("prix_fruits_pkg")
        if pkg is None:
            return 35.0 + regional_premium.get(localisation, 0)
        saison_export = 1 if mois in [3, 4, 5, 6] else 0
        base_lag = 28.0 + 4.0 * sin_mois  # varie entre 24 et 32
        features = np.array([[
            mois, annee, sin_mois, cos_mois, eur_mad, saison_export,
            0.04 + 0.02 * abs(cos_mois),
            base_lag,
            base_lag - 1.0,
            base_lag - 2.0,
            base_lag - 0.8,
            0.02 + 0.01 * saison_export,
        ]])
        try:
            prix_base = float(pkg["model"].predict(features)[0])
            return round(prix_base + regional_premium.get(localisation, 0), 2)
        except Exception as e:
            logger.warning(f"Prix fruits prediction failed: {e}")
            return round(35.0 + regional_premium.get(localisation, 0), 2)


#  4. PRODUCTION PREDICTION (25 features)
def _predict_production(form_data: dict, vpd: float, ndwi: float,
                        pluie: float, gamma: float) -> float:
    """
    Predict Rendement_kg_ha using the production XGBRegressor (25 features).
    The model was trained on:
      13 numeric base features + one-hot(Culture, Localisation, Type_Sol)
    """
    xgb_prod = _models.get("xgb_prod")
    if xgb_prod is None:
        return 15000.0 * float(form_data.get("superficie_ha", 1.0))

    superficie = float(form_data.get("superficie_ha", 1.0))
    experience = float(form_data.get("annees_experience", 5))
    culture = form_data.get("culture", "avocat")
    localisation = form_data.get("localisation", "Souss")
    type_sol = form_data.get("type_sol", "Limoneux")

    surface_barrage_norm = _LOC_BARRAGE_NORM.get(localisation, 0.4)
    surface_barrage_spec = surface_barrage_norm  # same source in notebook

    # Build base features
    base = {
        "Superficie_ha":                superficie,
        "Annees_Experience":            experience,
        "Gamma":                        gamma,
        "Irrigation":                   int(form_data.get("irrigation", 0)),
        "Solaire":                      int(form_data.get("solaire", 0)),
        "IoT_sondes":                   int(form_data.get("iot_sondes", 0)),
        "Filets_Serres":                int(form_data.get("filets_serres", 0)),
        "VPD_moyen_saison":             vpd,
        "Jours_stress_thermique":       max(0, int((vpd - 2.0) * 15)),
        "Pluie_saison_mm":              pluie,
        "NDWI_moyen":                   ndwi,
        "Surface_barrage_norm":         surface_barrage_norm,
        "Surface_barrage_specifique_norm": surface_barrage_spec,
    }

    # Build one-hot features
    onehot = {col: 0 for col in PROD_ONEHOT_COLS}
    culture_mapped = "Avocat" if culture.lower() == "avocat" else "Fruits_Rouges"
    culture_col = f"Culture_{culture_mapped}"
    if culture_col in onehot:
        onehot[culture_col] = 1

    loc_col = f"Localisation_{localisation}"
    if loc_col in onehot:
        onehot[loc_col] = 1

    sol_col = f"Type_Sol_{type_sol}"
    if sol_col in onehot:
        onehot[sol_col] = 1

    # Assemble in correct order
    row = [base[f] for f in PROD_BASE_FEATURES] + [onehot[f] for f in PROD_ONEHOT_COLS]
    X = np.array([row], dtype=float)

    try:
        rendement = float(xgb_prod.predict(X)[0])
        logger.info(f"Production predicted: {rendement:.0f} kg/ha")
        return rendement
    except Exception as e:
        logger.error(f"Production prediction failed: {e}")
        return 15000.0


#  5. CREDIT V6 DECISION (44 features)
def _build_44_features(form_data: dict, rendement: float, prix: float,
                       vpd: float, ndwi: float, pluie: float,
                       gamma: float) -> pd.DataFrame:
    """
    Build the 44-column DataFrame expected by XGBoost Credit V6.
    Includes 20 original numeric + 21 derived + 3 categorical columns.
    Formulas are from the training notebook (01_train_xgboost_v6.ipynb).
    """
    v6_pkg = _models.get("v6_pkg")
    if v6_pkg is None:
        return None

    feature_names = v6_pkg["feature_names"]
    superficie = float(form_data.get("superficie_ha", 1.0))
    experience = float(form_data.get("annees_experience", 5))
    culture = form_data.get("culture", "avocat")
    localisation = form_data.get("localisation", "Souss")
    type_sol = form_data.get("type_sol", "Limoneux")

    irrigation = int(form_data.get("irrigation", 0))
    solaire = int(form_data.get("solaire", 0))
    iot_sondes = int(form_data.get("iot_sondes", 0))
    filets_serres = int(form_data.get("filets_serres", 0))

    surface_barrage_norm = _LOC_BARRAGE_NORM.get(localisation, 0.4)
    jours_stress = max(0, int((vpd - 2.0) * 15))

    # Production and revenue estimates
    production_totale = rendement * superficie
    revenu_brut = production_totale * prix
    charges_totales = superficie * 25000  # ~25k MAD/ha typical cost
    revenu_net = revenu_brut - charges_totales

    # Risque brut (from notebook formula)
    risque_brut = round(max(0.5, min(3.0,
        1.0
        + (vpd - 1.0) * 0.3
        + jours_stress * 0.005
        - ndwi * 0.5
        - surface_barrage_norm * 0.1
    )), 3)

    # SABC v6 formula: clip((Prod*Prix)/(Risque*(1-Gamma+0.01)*100000), 0, 100)
    _base_score = (production_totale * prix) / (risque_brut * 100000)
    _bonus_gamma    = gamma * 8          # assurance = max +8 pts
    _bonus_tech     = score_technologie * 3  # chaque équipement = +3 pts
    _malus_exp      = max(0, (3 - experience) * 2)  # <3 ans = malus
    sabc_v6 = max(0, min(88,
    _base_score + _bonus_gamma + _bonus_tech - _malus_exp
))

    # ── 20 original numeric features ──
    num_orig = {
        "Superficie_ha":         superficie,
        "Annees_Experience":     experience,
        "Irrigation":            irrigation,
        "Solaire":               solaire,
        "IoT_sondes":            iot_sondes,
        "Filets_Serres":         filets_serres,
        "Gamma":                 gamma,
        "VPD_moyen_saison":      vpd,
        "Jours_stress_thermique": jours_stress,
        "Pluie_saison_mm":       pluie,
        "NDWI_moyen":            ndwi,
        "Surface_barrage_norm":  surface_barrage_norm,
        "Rendement_kg_ha":       rendement,
        "Production_totale_kg":  production_totale,
        "Prix_vente_MAD_kg":     prix,
        "Revenu_brut_MAD":       revenu_brut,
        "Charges_totales_MAD":   charges_totales,
        "Revenu_net_MAD":        revenu_net,
        "Risque_brut":           risque_brut,
        "sabc_v6":               sabc_v6,
    }

    # ── 21 derived features (formulas from training notebook) ──
    score_technologie = irrigation + solaire + iot_sondes + filets_serres

    num_derived = {
        "marge_nette_ratio":       revenu_net / (revenu_brut + 1),
        "liquidite_ratio":         revenu_brut / (charges_totales + 1),
        "charges_ratio":           charges_totales / (revenu_brut + 1),
        "couverture_charges":      revenu_brut / (charges_totales + 1),
        "revenu_net_par_ha":       revenu_net / (superficie + 0.01),
        "revenu_brut_par_ha":      revenu_brut / (superficie + 0.01),
        "seuil_rentabilite_prix":  charges_totales / (production_totale + 1),
        "marge_sur_prix":          prix - charges_totales / (production_totale + 1),
        "prix_par_rendement":      prix * rendement,
        "stress_hydrique_compose": vpd * jours_stress,
        "indice_eau_saison":       ndwi * pluie,
        "efficience_hydrique":     rendement / (pluie + 1),
        "score_technologie":       score_technologie,
        "technologie_experience":  score_technologie * experience,
        "technologie_superficie":  score_technologie * superficie,
        "sabc_risque_ratio":       sabc_v6 / (risque_brut + 0.01),
        "sabc_technologie":        sabc_v6 * score_technologie / 4.0,
        "sabc_log":                np.log1p(sabc_v6),
        "gamma_experience":        gamma * experience,
        "risque_superficie":       risque_brut * superficie,
        "ndwi_pluie":              ndwi * pluie,
    }

    # ── 3 categorical features ──
    culture_mapped = "Avocat" if culture.lower() == "avocat" else "Fruits_Rouges"
    cats = {
        "Culture":      culture_mapped,
        "Localisation": localisation,
        "Type_Sol":     type_sol,
    }

    # Assemble in feature_names order
    all_vals = {}
    all_vals.update(num_orig)
    all_vals.update(num_derived)
    all_vals.update(cats)

    df = pd.DataFrame([{fn: all_vals.get(fn, 0) for fn in feature_names}])

    # Apply ordinal encoding on categorical columns
    oe = v6_pkg["ordinal_encoder"]
    cat_columns = v6_pkg["cat_columns"]
    df[cat_columns] = oe.transform(df[cat_columns])

    return df


# ══════════════════════════════════════════════════════
# 6. MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════
def traiter_dossier_agriculteur(form_data: dict) -> dict:
    """
    Orchestrate the full backend pipeline and return the result dict.
    RAG advice is NOT included — it is called separately on demand.

    Returns:
        dict with keys: decision, probabilites, score_sabc,
                        equipements_manquants, rendement_estime,
                        prix_estime, production_estimee, revenu_brut,
                        vpd_j7, ndwi_j30, stress_thermique, stress_hydrique, risque_brut
    """
    v6_pkg = _models.get("v6_pkg")
    superficie = float(form_data.get("superficie_ha", 1.0))
    culture = form_data.get("culture", "avocat")
    
    # ⚠️ NOUVEAU : On récupère la localisation pour dynamiser le climat
    localisation = form_data.get("localisation", "Souss")

    # A. Missing equipment
    equipements_manquants = []
    if int(form_data.get("irrigation", 0)) == 0:
        equipements_manquants.append("Irrigation Goutte-a-goutte")
    if int(form_data.get("solaire", 0)) == 0:
        equipements_manquants.append("Panneaux solaires")
    if int(form_data.get("iot_sondes", 0)) == 0:
        equipements_manquants.append("Sondes meteo / IoT")
    if int(form_data.get("filets_serres", 0)) == 0:
        equipements_manquants.append("Filets / Serres")

    # B. Climate forecasts (Calculated from your LSTM & SARIMA engines)
    logger.info("Computing climate forecasts...")
    
    # ⚠️ NOUVEAU : On passe la localisation aux fonctions
    vpd_pred = _forecast_vpd(localisation)   
    ndwi_pred = _forecast_ndwi(localisation) 
    
    pluie = 200.0    # default seasonal rainfall (mm)
    gamma = float(form_data.get("aversion_risque", 0.5)) # Dynamisé avec le slider de app.py

    # C. Prix prediction (time-series model)
    logger.info("Predicting market prices...")
    prix_estime = _predict_prix(culture)

    # D. Production prediction (25-feature model)
    logger.info("Predicting production...")
    rendement_estime = _predict_production(form_data, vpd_pred, ndwi_pred, pluie, gamma)
    production_estimee = rendement_estime * superficie
    revenu_brut = production_estimee * prix_estime

    # E. Credit V6 decision (44-feature model)
    decision_ia = "CONDITIONNEL"
    probabilites = {"APPROUVE": 0.0, "CONDITIONNEL": 100.0, "REFUSE": 0.0}
    sabc_v6 = float(form_data.get("sabc_estime", 55))

    if v6_pkg is not None:
        logger.info("Running credit decision model...")
        df_features = _build_44_features(
            form_data, rendement_estime, prix_estime,
            vpd_pred, ndwi_pred, pluie, gamma,
        )
        if df_features is not None:
            try:
                model = v6_pkg["model"]
                probas = model.predict_proba(df_features)[0]
                classes = v6_pkg["classes"]
                probabilites = {
                    cls: round(float(p) * 100, 1)
                    for cls, p in zip(classes, probas)
                }
                decision_ia = classes[np.argmax(probas)]
                # Extract the computed SABC from features
                sabc_v6 = float(df_features.at[0, "sabc_v6"])
            except Exception as e:
                logger.error(f"Credit V6 prediction failed: {e}")

    # F. Rule-based agro-climatic signals deduction
    # Map the scalar outputs to categorical stress alerts for the UI
    stress_thermique = 1 if (vpd_pred is not None and vpd_pred > 2.5) else 0
    stress_hydrique = 1 if (ndwi_pred is not None and ndwi_pred < -0.15) else 0
    
    # Use the 'REFUSE' probability directly as a proxy for the raw risk score
    risque_brut = float(probabilites.get("REFUSE", 0.0)) / 100.0

    # ══════════════════════════════════════════════════════
    # RETURN DICTIONARY WITH ALL REQUIRED UI KEYS
    # ══════════════════════════════════════════════════════
    return {
        # Core Output
        "decision":              decision_ia,
        "probabilites":          probabilites,
        "score_sabc":           round(sabc_v6, 1),
        "equipements_manquants": equipements_manquants,
        
        # Intermediary Yield & Financial Estimates
        "rendement_estime":     round(rendement_estime, 0),
        "prix_estime":          round(prix_estime, 2),
        "production_estimee":   round(production_estimee, 0),
        "revenu_brut":          round(revenu_brut, 0),
        
        # Exposed climate & proxy keys to feed app.py seamlessly
        "vpd_j7":               float(vpd_pred) if vpd_pred is not None else None,
        "ndwi_j30":             float(ndwi_pred) if ndwi_pred is not None else None,
        "stress_thermique":     stress_thermique,
        "stress_hydrique":      stress_hydrique,
        "risque_brut":          risque_brut,
    }