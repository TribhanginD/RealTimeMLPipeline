"""
System health check for the Real-Time ML Pipeline.
Runs all components and reports PASS/FAIL for each.
"""
import subprocess, sys, os, json, time, traceback

ROOT = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline"
PYTHON = f"{ROOT}/.venv/bin/python3"
FEAST  = f"{ROOT}/.venv/bin/feast"

results = {}

def check(name, fn):
    try:
        fn()
        results[name] = "✅ PASS"
        print(f"  ✅  {name}")
    except Exception as e:
        results[name] = f"❌ FAIL — {e}"
        print(f"  ❌  {name}: {e}")

# ── 1. Data Generation ──────────────────────────────────────────────────────
def test_data_generation():
    import pandas as pd
    df = pd.read_parquet(f"{ROOT}/feature_store/data/transactions.parquet")
    required = {"user_id","amount","timestamp","is_fraud","latitude","longitude",
                "velocity_kmh","merchant_category","mean_amount_30d","std_amount_30d","z_score_amount"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert len(df) >= 1000, f"Too few rows: {len(df)}"
    assert df["is_fraud"].sum() > 0, "No fraud examples"

# ── 2. Feast Registry & Online Store ────────────────────────────────────────
def test_feast_online_store():
    sys.path.insert(0, ROOT)
    # Change to feature_store dir so Feast finds repo
    from feast import FeatureStore
    store = FeatureStore(repo_path=f"{ROOT}/feature_store")
    fv = store.list_feature_views()
    names = {f.name for f in fv}
    assert "transaction_stats" in names, "Missing transaction_stats feature view"
    assert "user_profile" in names,      "Missing user_profile feature view"
    # Try an online lookup
    vec = store.get_online_features(
        features=["transaction_stats:amount","user_profile:z_score_amount"],
        entity_rows=[{"user_id": "user_1"}],
    ).to_dict()
    assert "amount" in vec, "amount not in online features"

# ── 3. MLflow Model Registry ────────────────────────────────────────────────
def test_mlflow_registry():
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    registered = [m.name for m in client.search_registered_models()]
    assert "fraud_detection_champion_v2" in registered, \
        f"Champion v2 not registered. Found: {registered}"
    # Ensure at least one version exists
    versions = client.search_model_versions("name='fraud_detection_champion_v2'")
    assert len(versions) > 0, "No versions for champion v2"

# ── 4. Champion Model Loadable ───────────────────────────────────────────────
def test_model_loads():
    import mlflow.sklearn, numpy as np
    model = mlflow.sklearn.load_model("models:/fraud_detection_champion_v2/latest")
    row = np.zeros((1, 10))  # 10 features
    prob = model.predict_proba(row)[0][1]
    assert 0.0 <= prob <= 1.0, f"Bad probability: {prob}"

# ── 5. Streaming Simulator (short run) ──────────────────────────────────────
def test_streaming_simulator():
    proc = subprocess.Popen(
        [PYTHON, "-u", f"{ROOT}/data_pipeline/streaming_simulator.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    time.sleep(8)
    proc.terminate()
    output = proc.stdout.read().decode()
    assert "Pushed streaming update" in output or "Simulation update" in output or "Starting" in output, \
        f"Streaming not working. Output: {output[:300]}"

# ── 6. FastAPI Server & Inference ───────────────────────────────────────────
def test_fastapi_inference():
    import subprocess, json as _json, signal
    srv = subprocess.Popen(
        [f"{ROOT}/.venv/bin/uvicorn", "serving.main:app",
         "--host", "0.0.0.0", "--port", "8765"],
        cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(8)
    try:
        # Normal transaction
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:8765/predict",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"user_id": "user_5", "amount": 30.0,
                               "merchant_category": "grocery",
                               "current_lat": 40.71, "current_lon": -74.00})],
            capture_output=True, text=True, timeout=10
        )
        resp = _json.loads(r.stdout)
        assert "fraud_probability" in resp, f"Bad response: {resp}"

        # Fraud transaction (high velocity + luxury)
        r2 = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:8765/predict",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"user_id": "user_5", "amount": 1900.0,
                               "merchant_category": "luxury_retail",
                               "current_lat": 34.05, "current_lon": -118.24})],
            capture_output=True, text=True, timeout=10
        )
        resp2 = _json.loads(r2.stdout)
        assert resp2.get("is_fraud") == 1, f"Expected fraud=1, got: {resp2}"

        # Prometheus metrics endpoint
        r3 = subprocess.run(
            ["curl", "-s", "http://localhost:8765/metrics"],
            capture_output=True, text=True, timeout=5
        )
        assert "http_request" in r3.stdout, "Prometheus metrics endpoint not working"

    finally:
        srv.terminate()

# ── 7. Drift Monitoring Script ───────────────────────────────────────────────
def test_drift_monitoring():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "drift", f"{ROOT}/monitoring/drift_monitoring.py"
    )
    assert spec is not None, "drift_monitoring.py not found"
    # Just check the file parses cleanly without running it
    with open(f"{ROOT}/monitoring/drift_monitoring.py") as f:
        src = f.read()
    compile(src, "drift_monitoring.py", "exec")

# ── 8. Airflow DAG Syntax ────────────────────────────────────────────────────
def test_airflow_dag():
    with open(f"{ROOT}/airflow/retrain_dag.py") as f:
        src = f.read()
    compile(src, "retrain_dag.py", "exec")


if __name__ == "__main__":
    print("\n══════════════════════════════════════════")
    print("  Real-Time ML Pipeline — System Health Check")
    print("══════════════════════════════════════════\n")

    check("1. Data Generation (parquet schema)",  test_data_generation)
    check("2. Feast Online Store",                test_feast_online_store)
    check("3. MLflow Model Registry",             test_mlflow_registry)
    check("4. Champion Model Loadable",           test_model_loads)
    check("5. Streaming Simulator",               test_streaming_simulator)
    check("6. FastAPI Inference + Prometheus",    test_fastapi_inference)
    check("7. Drift Monitoring Script",           test_drift_monitoring)
    check("8. Airflow DAG Syntax",               test_airflow_dag)

    print("\n══════════════════════════════════════════")
    passed = sum(1 for v in results.values() if v.startswith("✅"))
    total  = len(results)
    print(f"  Results: {passed}/{total} checks passed")
    print("══════════════════════════════════════════\n")
    if passed < total:
        sys.exit(1)
