import pandas as pd
import time
import random
from datetime import datetime
from feast import FeatureStore

# Initialize Feast
store = FeatureStore(repo_path="/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/feature_store")

def simulate_streaming():
    print("Starting Streaming Feature Simulator...")
    user_ids = [f"user_{i}" for i in range(1, 101)]
    
    while True:
        # Simulate a burst of transactions
        user_id = random.choice(user_ids)
        amount = round(random.uniform(10.0, 1000.0), 2)
        count = random.randint(1, 50)
        avg_amt = round(random.uniform(20.0, 300.0), 2)
        
        # In a real system, this would be computed by a streaming engine like Flink
        # and then "pushed" to Feast.
        
        # MOCK: Push to Online Store
        # We use write_to_online_store to simulate the streaming update
        data_to_push = pd.DataFrame([{
            "user_id": user_id,
            "amount": amount,
            "transaction_count_last_24h": count,
            "avg_amount_last_24h": avg_amt,
            "timestamp": datetime.now()
        }])
        
        try:
            store.push("transaction_stats_push", data_to_push) 
            # Note: PushSource needs to be defined in Feast. 
            # For now, we can also use materialize_incremental or just simulate updates.
            # Let's update the feature definitions to include a PushSource for cleaner demo.
            print(f"Pushed streaming update for {user_id}: Amt={amount}, Count={count}")
        except Exception as e:
            # Fallback for demo: just print if PushSource isn't ready
            print(f"Simulation update for {user_id} (Internal push logic)")

        time.sleep(2) # Update every 2 seconds

if __name__ == "__main__":
    simulate_streaming()
