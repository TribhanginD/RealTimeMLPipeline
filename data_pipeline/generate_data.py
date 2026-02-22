import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math

# US city lat/long clusters to simulate realistic geo data
CITY_CLUSTERS = [
    {"name": "New York",     "lat": 40.71, "lon": -74.00},
    {"name": "Los Angeles",  "lat": 34.05, "lon": -118.24},
    {"name": "Chicago",      "lat": 41.87, "lon": -87.62},
    {"name": "Houston",      "lat": 29.76, "lon": -95.36},
    {"name": "Phoenix",      "lat": 33.44, "lon": -112.07},
    {"name": "Philadelphia", "lat": 39.95, "lon": -75.16},
    {"name": "San Antonio",  "lat": 29.42, "lon": -98.49},
    {"name": "San Diego",    "lat": 32.71, "lon": -117.15},
]

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "dining", "gas", "atm_withdrawal", "luxury_retail", "pharmacy"]

# High-risk categories inflate fraud probability
HIGH_RISK_CATEGORIES = {"atm_withdrawal", "luxury_retail", "electronics"}

def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two coords."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def generate_transaction_data(n=5000):
    user_ids = [f"user_{i}" for i in range(1, 101)]
    
    # Assign each user a home city
    user_home = {uid: random.choice(CITY_CLUSTERS) for uid in user_ids}
    # Track last transaction per user (for velocity)
    user_last_txn = {}

    data = []
    start_time = datetime.now()

    for i in range(n):
        user_id = random.choice(user_ids)
        merchant_cat = random.choice(MERCHANT_CATEGORIES)

        # Base fraud probability: ~5%, higher for risky categories
        base_fraud_prob = 0.05
        if merchant_cat in HIGH_RISK_CATEGORIES:
            base_fraud_prob = 0.12

        # Simulate geo: normal txns near home city, fraud may jump cities
        home = user_home[user_id]
        if random.random() < base_fraud_prob:
            # Fraudulent: pick a different city far away
            far_cities = [c for c in CITY_CLUSTERS if c["name"] != home["name"]]
            city = random.choice(far_cities)
            is_fraud = 1
        else:
            city = home
            is_fraud = 0

        # Add a small jitter around city center
        lat = city["lat"] + random.gauss(0, 0.1)
        lon = city["lon"] + random.gauss(0, 0.1)

        # Amount: fraud often involves higher amounts
        if is_fraud:
            amount = round(random.uniform(300.0, 2000.0), 2)
        else:
            amount = round(random.uniform(5.0, 300.0), 2)

        timestamp = start_time - timedelta(minutes=random.randint(0, 43200))  # last 30 days

        # Compute velocity: km/h vs. last transaction
        velocity_kmh = 0.0
        if user_id in user_last_txn:
            last = user_last_txn[user_id]
            dist_km = haversine(last["lat"], last["lon"], lat, lon)
            dt_hours = max(abs((timestamp - last["ts"]).total_seconds()) / 3600, 0.001)
            velocity_kmh = round(dist_km / dt_hours, 2)

        user_last_txn[user_id] = {"lat": lat, "lon": lon, "ts": timestamp}

        # Track user-level stats (rolling approximation for synthetic data)
        user_txns = [d["amount"] for d in data if d["user_id"] == user_id]
        mean_amount_30d = round(float(np.mean(user_txns)) if user_txns else amount, 2)
        std_amount_30d  = round(float(np.std(user_txns)) if len(user_txns) > 1 else 1.0, 2)
        z_score_amount  = round((amount - mean_amount_30d) / max(std_amount_30d, 0.01), 4)

        data.append({
            "user_id": user_id,
            "amount": amount,
            "timestamp": timestamp,
            "is_fraud": is_fraud,
            "transaction_count_last_24h": random.randint(1, 50),
            "avg_amount_last_24h": round(random.uniform(10, 300), 2),
            # New features
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "velocity_kmh": velocity_kmh,
            "merchant_category": merchant_cat,
            "mean_amount_30d": mean_amount_30d,
            "std_amount_30d": std_amount_30d,
            "z_score_amount": z_score_amount,
        })

    df = pd.DataFrame(data)
    output_path = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store/data/transactions.parquet"
    df.to_parquet(output_path)
    print(f"Generated {n} transactions with {df['is_fraud'].sum()} fraud cases ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_transaction_data(5000)
