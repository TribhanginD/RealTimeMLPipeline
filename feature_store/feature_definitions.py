from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Float64, Int64, String
from feast.value_type import ValueType

# Entity
user = Entity(name="user_id", value_type=ValueType.STRING, join_keys=["user_id"], description="User ID")

# Offline source (enriched parquet)
transaction_source = FileSource(
    path="/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store/data/transactions.parquet",
    event_timestamp_column="timestamp",
)

# PushSource for streaming updates
push_source = PushSource(
    name="transaction_stats_push",
    batch_source=transaction_source,
)

# --- Feature View 1: Core Transaction Stats (streaming-ready)
transaction_stats_view = FeatureView(
    name="transaction_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="amount",                    dtype=Float32),
        Field(name="transaction_count_last_24h", dtype=Int64),
        Field(name="avg_amount_last_24h",        dtype=Float32),
        # Geo features
        Field(name="latitude",                   dtype=Float64),
        Field(name="longitude",                  dtype=Float64),
        Field(name="velocity_kmh",               dtype=Float64),
        # Merchant
        Field(name="merchant_category",          dtype=String),
    ],
    online=True,
    source=push_source,
    tags={"team": "fraud_detection"},
)

# --- Feature View 2: User 30-Day Behavioural Profile
user_profile_view = FeatureView(
    name="user_profile",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="mean_amount_30d", dtype=Float32),
        Field(name="std_amount_30d",  dtype=Float32),
        Field(name="z_score_amount",  dtype=Float32),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "fraud_detection"},
)
