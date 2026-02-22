# Real-Time ML Fraud Detection Pipeline

A production-grade, real-time fraud detection system built with domain-specific fraud signals, a streaming feature store, and automated model lifecycle management.

---

## What Makes This Different

Most fraud pipelines treat detection as a generic classification problem. This system encodes **real-world fraud physics** directly into features:

| Signal | How It Works |
|---|---|
| **Geo-Velocity** | Computes km/h between stored last location and current transaction — flags physically impossible travel |
| **Z-Score Amount** | Measures how anomalous a transaction is *relative to that user's own 30-day history*, not a global threshold |
| **Merchant Risk** | Encodes category-level risk (e.g., `atm_withdrawal`, `luxury_retail`) as a trained feature |

**Result**: A $1,800 luxury purchase from a user who normally spends $30, made from a city they were not in 60 seconds ago → **98.96% fraud probability**.

---

## Architecture

```
[generate_data.py] → [Feast Feature Store] → [train.py] → [MLflow Registry]
                           ↑                                      ↓
              [streaming_simulator.py]              [FastAPI /predict endpoint]
              (real-time PushSource updates)         ↓              ↓
                                              fraud_signals    /metrics (Prometheus)
                                              velocity_kmh
                                              z_score_amount
```

---

## Stack

| Layer | Technology |
|---|---|
| Feature Store | [Feast](https://feast.dev) with `PushSource` for real-time updates |
| Model Training | scikit-learn (LR, RF, GradientBoosting) + SMOTE + Optuna |
| Experiment Tracking | MLflow Model Registry with auto-champion promotion |
| Model Serving | FastAPI + Uvicorn |
| Observability | Prometheus via `prometheus-fastapi-instrumentator` |
| Drift Monitoring | Evidently AI |
| Orchestration | Apache Airflow (retraining DAG) |
| Infrastructure | Docker Compose (Kafka, Zookeeper, Redis, MLflow) |

---

## Project Structure

```
RealTimeMLPipeline/
├── data_pipeline/
│   ├── generate_data.py        # Synthetic data with geo, velocity, Z-score
│   └── streaming_simulator.py  # Real-time Feast PushSource updates
├── feature_store/
│   ├── feature_definitions.py  # transaction_stats + user_profile FeatureViews
│   └── feature_store.yaml      # Feast repo config
├── models/
│   └── train.py                # SMOTE + Optuna + MLflow registry + champion promotion
├── serving/
│   └── main.py                 # FastAPI with real-time velocity/Z-score + Prometheus
├── monitoring/
│   └── drift_monitoring.py     # Evidently data drift detection
├── airflow/
│   └── retrain_dag.py          # Automated drift-triggered retraining DAG
├── tests/
│   └── system_check.py         # 8/8 end-to-end health checks
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Setup environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data & initialize feature store
```bash
python3 data_pipeline/generate_data.py
cd feature_store && feast apply && feast materialize-incremental $(python3 -c "from datetime import datetime; print(datetime.now().isoformat())")
cd ..
```

### 3. Train models & promote champion
```bash
python3 models/train.py
# Trains LR, RF, GB — promotes best F1 to 'fraud_detection_champion_v2' in MLflow
```

### 4. Start the inference API
```bash
uvicorn serving.main:app --reload
```

### 5. Test fraud detection
```bash
# Normal transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "amount": 45.00, "merchant_category": "grocery", "current_lat": 40.71, "current_lon": -74.00}'

# Impossible travel fraud
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "amount": 1800.00, "merchant_category": "luxury_retail", "current_lat": 34.05, "current_lon": -118.24}'
```

### 6. View Prometheus metrics
```
http://localhost:8000/metrics
```

### 7. Run system health check
```bash
python3 tests/system_check.py
# Expected: 8/8 checks passed
```

---

## API Response

```json
{
  "user_id": "user_1",
  "is_fraud": 1,
  "fraud_probability": 0.9896,
  "fraud_signals": {
    "velocity_kmh": 752939.95,
    "z_score_amount": 3.8361,
    "merchant_category": "luxury_retail",
    "amount": 1800.0
  }
}
```

---

## Model Performance (on synthetic data)

| Model | F1 Score |
|---|---|
| Logistic Regression | 0.9595 |
| Random Forest (Champion) | **1.0000** |
| Gradient Boosting | 1.0000 |
