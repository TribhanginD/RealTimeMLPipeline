from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append("/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fraud_model_retraining',
    default_args=default_args,
    description='Automated retraining of the fraud detection models',
    schedule_interval=timedelta(days=1),
)

def run_retraining():
    # Use the existing training script
    venv_python = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/.venv/bin/python"
    train_script = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/models/train.py"
    os.system(f"{venv_python} {train_script}")

def run_drift_check():
    venv_python = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/.venv/bin/python"
    drift_script = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/monitoring/drift_monitoring.py"
    # Execute drift check and return status
    result = os.system(f"{venv_python} {drift_script}")
    if result != 0:
        print("ALERT: Data drift detected! Proceeding to retraining.")
    else:
        print("No significant drift detected. Retraining as per schedule.")

check_drift = PythonOperator(
    task_id='check_data_drift',
    python_callable=run_drift_check,
    dag=dag,
)

retrain_models = PythonOperator(
    task_id='retrain_models',
    python_callable=run_retraining,
    dag=dag,
)

check_drift >> retrain_models
