import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from feast import FeatureStore
import os

# Initialize Feast
store = FeatureStore(repo_path="/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store")

def monitor_drift():
    # 1. Get reference data (training data)
    transactions_df = pd.read_parquet("/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store/data/transactions.parquet")
    entity_df = transactions_df[["user_id", "timestamp"]].iloc[:1000] # First 1000 as reference
    
    reference_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:amount",
            "transaction_stats:transaction_count_last_24h",
            "transaction_stats:avg_amount_last_24h",
        ],
    ).to_df().drop(columns=["user_id", "timestamp"])
    
    # 2. Get current data (simulated as the rest of the data or new data)
    current_entity_df = transactions_df[["user_id", "timestamp"]].iloc[1000:2000]
    current_data = store.get_historical_features(
        entity_df=current_entity_df,
        features=[
            "transaction_stats:amount",
            "transaction_stats:transaction_count_last_24h",
            "transaction_stats:avg_amount_last_24h",
        ],
    ).to_df().drop(columns=["user_id", "timestamp"])
    
    # 3. Generate Drift Report
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save report
    os.makedirs("/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/monitoring/reports", exist_ok=True)
    report.save_html("/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/monitoring/reports/drift_report.html")
    print("Drift report generated: monitoring/reports/drift_report.html")

if __name__ == "__main__":
    monitor_drift()
