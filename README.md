# AQI Predictor

A fully serverless, end-to-end machine learning system that forecasts Air Quality Index (AQI) three days ahead for three Pakistani cities — **Bahawalpur**, **Lahore**, and **Islamabad** — built as part of the 10 Pearls Data Science Internship ("Shine").

**Live dashboard:** https://aqi-predictor-nslt99a5awaqkn4fypvqen.streamlit.app/
**Repository:** https://github.com/UshbaJ/aqi-predictor

---

## What it does

- Pulls hourly air-quality and weather data automatically, every hour, for all three cities
- Forecasts **daily-average AQI** for the next 3 days (Day 1, Day 2, Day 3) using Ridge regression
- Retrains all models automatically once a day on the latest data
- Explains every prediction with SHAP feature importance
- Validates itself honestly with a genuine 90-day out-of-sample holdout (not just training-set metrics)
- Surfaces hazardous air-quality alerts, a what-if simulator, a voice briefing, and a personal exposure calculator
- Compares pollution patterns and levels across all three cities

## Tech stack

| Purpose | Tool |
|---|---|
| Data sources | OpenWeather Air Pollution API, Open-Meteo Archive API |
| Feature store | Hopsworks (Serverless) |
| Modeling | scikit-learn (Ridge, Random Forest) |
| Explainability | SHAP |
| Automation / CI-CD | GitHub Actions |
| Dashboard | Streamlit (Community Cloud) |
| Language | Python 3.13 |

## Project structure

```
aqi-predictor/
├── src/
│   ├── data_collection.py      # Pulls AQI + weather data from OpenWeather / Open-Meteo, per city
│   ├── features.py             # Feature engineering, Hopsworks read/write, city-aware loading
│   ├── feature_pipeline.py     # Pushes engineered features to Hopsworks Feature Store, per city
│   ├── train.py                # 5-fold TimeSeriesSplit CV, trains + saves Ridge/RF models, per city
│   ├── predict.py              # Loads saved models, produces 3-day forecast, per city
│   ├── validate_holdout.py     # Genuine 90-day out-of-sample validation, per city
│   ├── compute_shap.py         # SHAP feature importance per horizon, per city
│   ├── app.py                  # Streamlit dashboard
│   └── *.pkl                   # Trained Ridge models (Random Forest models are gitignored — large, unused in production)
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory analysis for all 3 cities + cross-city comparison
├── data/
│   ├── {city}/raw_aqi_data.csv, raw_weather_data.csv   # Raw pulls (gitignored — Hopsworks is the source of truth)
│   ├── cv_validation_results*.csv
│   ├── holdout_*.csv
│   └── shap_importance_*.csv
├── .github/workflows/
│   ├── feature-pipeline.yml    # Hourly: pulls fresh data + pushes to Hopsworks, matrixed across all 3 cities
│   └── daily-retrain.yml       # Daily: retrains all models, matrixed across all 3 cities
├── .streamlit/
│   └── config.toml             # Forces light theme (fixes dark-mode inheritance from visitor browsers)
├── requirements.txt             # Full dashboard dependencies
├── requirements-pipeline.txt    # Minimal dependencies for CI (avoids a streamlit/hopsworks protobuf conflict)
└── runtime.txt                  # Pins Python 3.13 for Streamlit Cloud (hopsworks requires <3.14)
```

## How the pipeline works

1. **Hourly Feature Pipeline** (GitHub Actions, every hour, all 3 cities in parallel)
   `data_collection.py --city <city> --hours-back 48` → `feature_pipeline.py --city <city>` → Hopsworks Feature Store

2. **Daily Retraining** (GitHub Actions, once a day, all 3 cities in parallel)
   `train.py --city <city>` reads from Hopsworks (retrying on transient failures, falling back to local CSVs, and skipping cleanly rather than crashing if both are unavailable) → saves updated Ridge + Random Forest models for all 3 horizons

3. **Dashboard** (Streamlit Cloud)
   Reads the latest features and saved models live; users pick a city from a dropdown and see current conditions, a 3-day forecast, model validation, SHAP explanations, and cross-city comparisons.

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# One-time historical backfill (per city)
python src/data_collection.py --city bahawalpur --backfill
python src/data_collection.py --city lahore --backfill
python src/data_collection.py --city islamabad --backfill

# Push features to Hopsworks (per city)
python src/feature_pipeline.py --city bahawalpur
python src/feature_pipeline.py --city lahore
python src/feature_pipeline.py --city islamabad

# Train models (per city)
python src/train.py --city bahawalpur
python src/train.py --city lahore
python src/train.py --city islamabad

# Validation + SHAP (per city)
python src/validate_holdout.py --city bahawalpur
python src/compute_shap.py --city bahawalpur
# (repeat for lahore, islamabad)

# Run the dashboard
streamlit run src/app.py
```

Requires a `.env` file with:
```
OPENWEATHER_API_KEY=your_key
HOPSWORKS_API_KEY=your_key
```

## Model performance (Bahawalpur, 5-fold time-series CV)

| Horizon | Best model | Improvement over naive persistence baseline |
|---|---|---|
| +24h | Ridge | 18.4% lower RMSE |
| +48h | Ridge | 18.9% lower RMSE |
| +72h | Ridge | 20.5% lower RMSE |

Ridge consistently outperformed both the naive baseline and Random Forest, with lower variance across folds. Full per-city results are in `data/cv_validation_results*.csv` and rendered live on the dashboard's "Model & Validation" tab.

## Key design decisions

- **Targets are 24-hour window averages**, not point-in-time values — this matches the actual project specification (confirmed with mentors) rather than the initially-built point-in-time design.
- **Ridge over Random Forest**: simpler, more stable, and won on every horizon during proper time-series cross-validation.
- **Hopsworks as the single source of truth** for features; local CSVs exist only for local development and as an emergency fallback, never committed to the repository.
- **Multi-city support** was added after the single-city pipeline was fully validated and stable, following the mentor-recommended approach of building a working simple system before adding complexity.

## Known limitations

- Lahore and Islamabad have a shorter effective "settled" history in the feature store than Bahawalpur, and their holdout/CV metrics show higher variance — forecasts for these cities, especially at 72h, can overshoot during rapid pollution swings (Ridge is a linear model and extrapolates trends).
- Model retraining depends on Hopsworks availability; a multi-day Hopsworks outage would mean stale models until connectivity is restored (the pipeline fails safely — it skips retraining rather than crashing — but does not yet send an alert).

## Acknowledgements

Built under the 10 Pearls Data Science Shine Internship
