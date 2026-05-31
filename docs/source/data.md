# Data Sources and Format

## Primary Dataset

### flux_Ct_Ht_features_v4.csv

The main dataset synthesizes field observations, satellite extracts, and market prices across Moroccan agricultural zones (Souss-Massa, Gharb, Loukkos, Haouz, Tadla) from 2022 to 2025.

**Location:** `AvoRuby_Back/processed/flux_Ct_Ht_features_v4.csv`  
**Format:** CSV, comma-separated, UTF-8  
**Shape:** ~12,000 rows × 44 feature columns

#### Key Columns

| Column | Type | Description |
|---|---|---|
| `Culture` | categorical | `Avocat` or `Fruits_Rouges` |
| `Localisation` | categorical | Moroccan agricultural region |
| `Type_Sol` | categorical | Limoneux, Argileux, Sableux |
| `Superficie_ha` | float | Farm size in hectares |
| `Annees_Experience` | int | Grower years of experience |
| `VPD_moyen_saison` | float | Seasonal mean VPD (kPa) |
| `Jours_stress_thermique` | int | Days with VPD > 2 kPa |
| `Pluie_saison_mm` | float | Seasonal rainfall (mm) |
| `NDWI_moyen` | float | Mean NDWI from Sentinel-2 |
| `Surface_barrage_norm` | float | Normalized dam water storage |
| `Irrigation` | binary | 1 if drip irrigation installed |
| `Solaire` | binary | 1 if solar pumping installed |
| `IoT_sondes` | binary | 1 if IoT sensors installed |
| `Filets_Serres` | binary | 1 if anti-hail nets/greenhouses installed |
| `Gamma` | float | MAMDA insurance coverage (0–1) |
| `Rendement_kg_ha` | float | Target: actual yield (kg/ha) |
| `Prix_MAD_kg` | float | Target: actual market price |
| `Decision_Credit` | string | Target: APPROUVE / CONDITIONNEL / REFUSE |

## Market Data

### flux_marche.csv

Weekly market price time-series for avocado and red fruits in Moroccan wholesale markets (Casablanca, Agadir, Kenitra) from 2022 to 2025.

**Columns:**  
- `Date` — Week starting date (YYYY-MM-DD)
- `Culture` — `Avocat` or `Fruits_Rouges`
- `Prix_MAD_kg` — Market price (MAD/kg)
- `Marche` — Market name
- `Volume_tonnes` — Weekly trading volume

## Resilience Data

### flux_resilience.csv

Seasonal resilience indices derived from Sentinel-2 NDWI, MODIS LST, and ground truth yield records collected between 2022 and 2025.

**Columns:**  
- `Date` — Season start date
- `Zone` — Agricultural zone
- `NDWI_moyen` — Mean NDWI
- `VPD_moyen` — Mean VPD
- `Gamma` — Insurance coverage
- `Rendement_reel` — Actual yield
- `Risque_brut` — Raw risk indicator

## Climate Data Provenance

| Indicator | Source | Spatial Resolution | Period |
|---|---|---|---|
| NDWI | Sentinel-2 (ESA Copernicus) | 10 m, 10-day | 2022–2025 |
| VPD | ERA5-Land Reanalysis (ECMWF) | 9 km, hourly | 2022–2025 |
| Rainfall | CHIRPS v2.0 (UCSB/USGS) | 5 km, daily | 2022–2025 |
| Dam level | ABHSM (Bassin Hydraulique) | Monthly | 2022–2025 |
| Temperature | ERA5-Land Reanalysis (ECMWF) | 9 km, hourly | 2022–2025 |

## Data Format & Encoding

- **Encoding:** UTF-8
- **Decimal separator:** `.` (period)
- **Date format:** ISO 8601 (YYYY-MM-DD)
- **Missing values:** `NaN` (represented as empty in CSV)
- **Categorical encoding:** String values (no numeric codes)

## Data Quality Notes

- All datasets have been validated for missing values and outliers
- Sentinel-2 data gaps filled using linear interpolation where gaps < 30 days
- ERA5-Land data downscaled using bilinear interpolation to 5 km resolution
- Market prices adjusted for inflation using CPI baseline 2020
