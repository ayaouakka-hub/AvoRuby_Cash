# Results and Validation

All metrics reported here are extracted directly from the training notebooks
(`01_train_xgboost_v6.ipynb` and `04_modelisation.ipynb`).

---

## 1. VPD Forecasting — Model Comparison

Three models were trained and evaluated for 7-day VPD forecasting (J+7):

| Model | MAE (kPa) | RMSE (kPa) | R² | MAPE | Pearson r |
|---|---|---|---|---|---|
| SARIMA (weekly) | 0.1966 | 0.2575 | 0.9290 | 17.45% | 0.9685 |
| XGBoost (daily) | 0.1915 | 0.2504 | 0.9344 | 17.20% | 0.9680 |
| **LSTM PyTorch (daily)** | **0.1728** | **0.2376** | **0.9448** | **13.62%** | **0.9727** |

**Selected model: LSTM PyTorch** — best R² (0.9448) and lowest MAE (0.1728 kPa).

![VPD Model Comparison](AvoRuby_Back/processed/comparaison_vpd_3modeles.png)

![LSTM Loss Curve](AvoRuby_Back/processed/lstm_loss.png)

![LSTM VPD Results](AvoRuby_Back/processed/lstm_vpd_results.png)

---

## 2. NDWI Forecasting — SARIMA

Monthly NDWI forecasting (J+30) using SARIMA(1,1,1)(1,1,1)[12]:

| Metric | Value |
|---|---|
| MAE | 0.0140 |
| RMSE | 0.0174 |
| R² | 0.3082 |
| MAPE | 14.78% |
| Pearson r | 0.6684 |

> **Note:** The moderate R² reflects the inherent variability of satellite-derived NDWI
> at monthly resolution. The MAE of 0.014 is agronomically acceptable for stress detection.

![SARIMA NDWI](AvoRuby_Back/processed/sarima_ndwi.png)

---

## 3. Market Price Prediction — XGBoost

Two separate XGBoost regressors trained for avocado and red fruits:

| Culture | R² | MAE (MAD/kg) |
|---|---|---|
| Avocado | 0.7324 | 2.75 |
| Red Fruits | 0.7243 | 7.07 |

![XGBoost Price](AvoRuby_Back/processed/xgboost_prix.png)

---

## 4. Yield Production — XGBoost

| Metric | Value |
|---|---|
| MAE | 1,287 kg/ha |
| RMSE | 1,617 kg/ha |
| R² | 0.5708 |
| MAPE | 13.22% |
| Pearson r | 0.7623 |

![XGBoost Production](AvoRuby_Back/processed/xgboost_production.png)

---

## 5. Credit Decision — XGBoost v6 vs v5

The v6 model shows a major improvement over v5 thanks to SMOTE-NC rebalancing
and 44 engineered features:

| Metric | XGBoost v5 | XGBoost v6 |
|---|---|---|
| Accuracy | 0.683 | **0.983** |
| F1 macro | 0.587 | **0.971** |
| F1 weighted | 0.710 | **0.984** |
| AUC-ROC | — | **1.000** |

### Per-Class Performance (v6)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| APPROUVE | 1.000 | 0.960 | 0.980 |
| CONDITIONNEL | 0.875 | 1.000 | 0.933 |
| REFUSE | 1.000 | 1.000 | 1.000 |

### Brier Scores (v6) — closer to 0 is better

| Class | Brier Score |
|---|---|
| APPROUVE | 0.0109 — Excellent |
| CONDITIONNEL | 0.0108 — Excellent |
| REFUSE | 0.0000 — Perfect |

### Top 10 Most Important Features (v6)

| Rank | Feature | Importance |
|---|---|---|
| 1 | sabc_log | 0.305 |
| 2 | sabc_risque_ratio | 0.292 |
| 3 | sabc_v6 | 0.285 |
| 4 | Risque_brut | 0.035 |
| 5 | sabc_technologie | 0.025 |
| 6 | Revenu_brut_MAD | 0.025 |
| 7 | indice_eau_saison | 0.009 |
| 8 | Production_totale_kg | 0.007 |
| 9 | Revenu_net_MAD | 0.005 |
| 10 | VPD_moyen_saison | 0.004 |

![Model Comparison v5 vs v6](AvoRuby_Back/processed/comparaison_modeles.png)

![Confusion Matrix v6](AvoRuby_Back/processed/confusion_matrix_v6.png)

![Feature Importance v6](AvoRuby_Back/processed/feature_importance_v6.png)

![Calibration Curves v6](AvoRuby_Back/processed/calibration_curves_v6.png)

![SHAP Global Importance](AvoRuby_Back/processed/shap_importance.png)

![SABC Score Distribution](AvoRuby_Back/processed/sabc_final.png)

![Risk Distribution](AvoRuby_Back/processed/distribution_risque_brut.png)
