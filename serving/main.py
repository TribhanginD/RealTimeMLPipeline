from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from feast import FeatureStore
import os
import math
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Fraud Detection Inference Service v2")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Feature Store
store = FeatureStore(repo_path="/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store")

# Merchant category encoder (same as training)
MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "dining", "gas", "atm_withdrawal", "luxury_retail", "pharmacy"]
le = LabelEncoder()
le.fit(MERCHANT_CATEGORIES)

FEATURE_COLS = [
    "amount",
    "transaction_count_last_24h",
    "avg_amount_last_24h",
    "latitude",
    "longitude",
    "velocity_kmh",
    "merchant_category_enc",
    "mean_amount_30d",
    "std_amount_30d",
    "z_score_amount",
]

def haversine(lat1, lon1, lat2, lon2):
    """Returns great-circle distance in km between two coordinates."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

class PredictRequest(BaseModel):
    user_id: str
    # Optional: caller can supply current transaction context
    amount: float | None = None
    merchant_category: str | None = None
    current_lat: float | None = None
    current_lon: float | None = None

@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Retrieve online features from Feast
    feature_vector = store.get_online_features(
        features=[
            "transaction_stats:amount",
            "transaction_stats:transaction_count_last_24h",
            "transaction_stats:avg_amount_last_24h",
            "transaction_stats:latitude",
            "transaction_stats:longitude",
            "transaction_stats:velocity_kmh",
            "transaction_stats:merchant_category",
            "user_profile:mean_amount_30d",
            "user_profile:std_amount_30d",
            "user_profile:z_score_amount",
        ],
        entity_rows=[{"user_id": request.user_id}],
    ).to_dict()

    # Check if user exists in online store
    if feature_vector["amount"][0] is None:
        return {"user_id": request.user_id, "prediction": "unknown", "reason": "user_not_found_in_online_store"}

    # 2. Override with request-level context if provided
    amount = request.amount if request.amount is not None else feature_vector["amount"][0]
    merchant_cat = request.merchant_category if request.merchant_category else (feature_vector["merchant_category"][0] or "grocery")
    stored_lat = feature_vector["latitude"][0] or 0.0
    stored_lon = feature_vector["longitude"][0] or 0.0

    # 3. Compute geo-velocity if new coordinates are provided
    velocity_kmh = feature_vector["velocity_kmh"][0] or 0.0
    if request.current_lat is not None and request.current_lon is not None:
        dist_km = haversine(stored_lat, stored_lon, request.current_lat, request.current_lon)
        velocity_kmh = dist_km / 0.0167  # assume ~1 min since last txn (worst case)

    # 4. Compute Z-score for this transaction
    mean_amt = feature_vector["mean_amount_30d"][0] or amount
    std_amt = max(feature_vector["std_amount_30d"][0] or 1.0, 0.01)
    z_score = (amount - mean_amt) / std_amt

    # 5. Build feature row
    merchant_enc = le.transform([merchant_cat])[0] if merchant_cat in le.classes_ else 0
    df = pd.DataFrame([{
        "amount": amount,
        "transaction_count_last_24h": feature_vector["transaction_count_last_24h"][0] or 0,
        "avg_amount_last_24h": feature_vector["avg_amount_last_24h"][0] or amount,
        "latitude": request.current_lat if request.current_lat is not None else stored_lat,
        "longitude": request.current_lon if request.current_lon is not None else stored_lon,
        "velocity_kmh": velocity_kmh,
        "merchant_category_enc": merchant_enc,
        "mean_amount_30d": mean_amt,
        "std_amount_30d": std_amt,
        "z_score_amount": z_score,
    }])

    # 6. Load champion v2 model from registry
    try:
        model = mlflow.sklearn.load_model("models:/fraud_detection_champion_v2/latest")
        prediction    = int(model.predict(df[FEATURE_COLS])[0])
        fraud_prob    = float(model.predict_proba(df[FEATURE_COLS])[0][1])
    except Exception as e:
        return {"error": f"Model load failed: {str(e)}"}

    return {
        "user_id": request.user_id,
        "is_fraud": prediction,
        "fraud_probability": round(fraud_prob, 4),
        "fraud_signals": {
            "velocity_kmh": round(velocity_kmh, 2),
            "z_score_amount": round(z_score, 4),
            "merchant_category": merchant_cat,
            "amount": amount,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
