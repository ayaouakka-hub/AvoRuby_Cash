# Methodology

The inference pipeline is orchestrated in `AvoRuby_Back/src/models.py` and follows a strict sequential dependency chain: **climate indicators → price → yield → credit decision**.

## 1. VPD Forecasting — LSTM

**Notebook:** `04_modelisation.ipynb`  
**Model:** `lstm_vpd.pt` + `scaler_X_lstm.pkl` + `scaler_y_lstm.pkl`

A bidirectional LSTM network trained on 10 years of hourly VPD time-series (ERA5-Land) to forecast the seasonal mean VPD from a 12-month input window.

**Architecture:**
- Input: 12-month sequence of [VPD, T_max, RH, Wind]
- 2 LSTM layers (128 hidden units, dropout 0.2)
- Dense output → single VPD forecast value

**Performance:**

| Metric | Value |
|---|---|
| RMSE | 0.043 kPa |
| MAE | 0.031 kPa |
| R² | 0.94 |

**Visualizations:**

![LSTM Training Loss](AvoRuby_Back/processed/lstm_loss.png)

![VPD 3-Model Comparison](AvoRuby_Back/processed/comparaison_vpd_3modeles.png)

![LSTM VPD Results](AvoRuby_Back/processed/lstm_vpd_results.png)

![SHAP VPD Importance](AvoRuby_Back/processed/shap_vpd.png)

## 2. NDWI Forecasting — SARIMA

**Notebook:** `04_modelisation.ipynb`  
**Model:** `sarima_ndwi.pkl`

A SARIMA(1,1,1)(1,1,1)[12] model fitted on 6 years of monthly NDWI time-series from Sentinel-2 mosaics. Model order selected via AIC/BIC grid search.

**Visualizations:**

![SARIMA NDWI Forecast](AvoRuby_Back/processed/sarima_ndwi.png)

## 3. Market Price Prediction — XGBoost

**Notebook:** `01_collecte_marche.ipynb`  
**Models:** `xgboost_prix_avocat.pkl`, `xgboost_prix_fruits_rouges.pkl`

Two XGBoost regressors trained on the `flux_marche.csv` time-series with calendar, lag, and macroeconomic features.

**Key Engineered Features:**
- Month, week-of-year, year
- Lag-1, Lag-4, Lag-12 monthly price lags
- Rolling 3-month mean and standard deviation
- Export competitor season flag (Spain/Poland)
- Regional rainfall lag (supply shock proxy)

**Visualization:**

![XGBoost Market Price](AvoRuby_Back/processed/xgboost_prix.png)

## 4. Yield Production Estimation — XGBoost

**Notebook:** `04_modelisation.ipynb`  
**Model:** `xgboost_production.pkl`

XGBoost regressor with 25 input features (after one-hot encoding) predicting yield (kg/ha) from agronomic profile, climate context, and technology level.

**Feature Groups:**
- **Agronomic:** `Superficie_ha`, `Annees_Experience`, `Culture`, `Localisation`, `Type_Sol`
- **Climatic:** `VPD_moyen_saison`, `Jours_stress_thermique`, `Pluie_saison_mm`, `NDWI_moyen`
- **Hydrological:** `Surface_barrage_norm`, `Surface_barrage_specifique_norm`
- **Equipment:** `Irrigation`, `Solaire`, `IoT_sondes`, `Filets_Serres`
- **Risk:** `Gamma` (insurance coverage factor)

**Visualizations:**

![XGBoost Production Yield](AvoRuby_Back/processed/xgboost_production.png)

![Gamma Flux Visualization](AvoRuby_Back/processed/visualisation_flux_Gamma.png)

![Ct/Ht Feature Visualization](AvoRuby_Back/processed/visualisation_Ct_Ht.png)

## 5. Credit Decision Model v6 — XGBoost (44 features)

**Notebook:** `01_train_xgboost_v6.ipynb`  
**Model:** `xgboost_credit_v6.pkl`

The core scoring model classifies each application into: **APPROUVE**, **CONDITIONNEL**, or **REFUSE**.

**Composite Features:**

| Feature | Description |
|---|---|
| `sabc_v6` | Composite SABC score (weighted agronomic/climatic/equipment criteria) |
| `revenu_brut_estime` | superficie × rendement × prix |
| `ratio_risque` | VPD × (1 - NDWI) — climate pressure index |
| `resilience_score` | Gamma × (1 - stress_ratio) |
| `tech_score` | Sum of equipment indicators (0–4) |
| `experience_normalise` | Log-transformed experience years |
| `barrage_pression` | Normalized dam level (water access security) |

**Class Imbalance Handling:** SMOTE-NC (Synthetic Minority Oversampling for Numerical and Categorical features) was applied to the original 72% / 18% / 10% class distribution.

**Visualizations:**

![SMOTE-NC Rebalancing](AvoRuby_Back/processed/smotenc_rebalancing_v6.png)

![Class Distribution v6](AvoRuby_Back/processed/class_distribution_v6.png)

![Confusion Matrix v6](AvoRuby_Back/processed/confusion_matrix_v6.png)

![Feature Importance v6](AvoRuby_Back/processed/feature_importance_v6.png)

![Feature Distributions by Class](AvoRuby_Back/processed/feature_distributions_v6.png)

![Probability Distributions v6](AvoRuby_Back/processed/proba_distributions_v6.png)

![Calibration Curves v6](AvoRuby_Back/processed/calibration_curves_v6.png)

![SHAP Global Importance](AvoRuby_Back/processed/shap_importance.png)

![Model Comparison v5 vs v6](AvoRuby_Back/processed/comparaison_modeles.png)
