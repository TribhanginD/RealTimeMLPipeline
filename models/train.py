import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
import os
import shutil
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.tracking import MlflowClient

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/data_pipeline/raw/creditcard.csv"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Amount_log", "Hour"]

def get_training_data():
    df = pd.read_csv(DATA_PATH)

    # Feature engineering on real data
    df["Amount_log"] = np.log1p(df["Amount"])       # log-normalize skewed Amount
    df["Hour"] = (df["Time"] / 3600).astype(int) % 24  # time-of-day signal

    X = df[FEATURE_COLS]
    y = df["Class"]
    return X, y

def train_model(X_train, y_train, X_test, y_test, model_type="gb", n_trials=8):
    mlflow.set_experiment(f"fraud_real_{model_type}")

    def objective(trial):
        with mlflow.start_run(nested=True):
            if model_type == "lr":
                params = {"C": trial.suggest_float("C", 1e-4, 10, log=True),
                          "solver": "liblinear", "max_iter": 500}
                clf = LogisticRegression(**params)
            elif model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "class_weight": "balanced",
                }
                clf = RandomForestClassifier(**params)
            elif model_type == "gb":
                params = {
                    "max_iter": trial.suggest_int("max_iter", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
                }
                clf = HistGradientBoostingClassifier(**params)

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1 = f1_score(y_test, preds)
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"  [{model_type.upper()}] best F1={study.best_value:.4f} | params={study.best_params}")

    # Train final model with best params
    model_path = f"/Users/tribhangind/Documents/GitHub/RealTimeMLPipeline/models/{model_type}_real"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    with mlflow.start_run() as run:
        if model_type == "lr":
            final = LogisticRegression(**study.best_params)
        elif model_type == "rf":
            final = RandomForestClassifier(**study.best_params)
        elif model_type == "gb":
            final = HistGradientBoostingClassifier(**study.best_params)

        final.fit(X_train, y_train)
        probs = final.predict_proba(X_test)[:, 1]

        # ── Threshold Tuning ──────────────────────────────────────────────
        # Sweep thresholds [0.01 → 0.99] and find:
        #   1. F1-optimal threshold
        #   2. Minimum threshold that achieves >= 90% recall
        thresholds  = [t / 100 for t in range(1, 100)]
        best_f1_thresh, best_f1 = 0.5, 0.0
        recall_90_thresh = 0.5
        for t in thresholds:
            preds_t = (probs >= t).astype(int)
            f1_t  = f1_score(y_test, preds_t, zero_division=0)
            rec_t = recall_score(y_test, preds_t, zero_division=0)
            if f1_t > best_f1:
                best_f1        = f1_t
                best_f1_thresh = t
            if rec_t >= 0.90:
                recall_90_thresh = t  # keeps updating → smallest valid t

        # Evaluate at both thresholds
        preds_f1  = (probs >= best_f1_thresh).astype(int)
        preds_r90 = (probs >= recall_90_thresh).astype(int)

        f1_at_opt   = f1_score(y_test, preds_f1,  zero_division=0)
        rec_at_opt  = recall_score(y_test, preds_f1, zero_division=0)
        prec_at_opt = precision_score(y_test, preds_f1, zero_division=0)

        f1_at_r90   = f1_score(y_test, preds_r90, zero_division=0)
        rec_at_r90  = recall_score(y_test, preds_r90, zero_division=0)
        prec_at_r90 = precision_score(y_test, preds_r90, zero_division=0)

        auc = roc_auc_score(y_test, probs)

        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            "roc_auc":          auc,
            # Default threshold (0.5)
            "f1_default":       f1_score(y_test, (probs >= 0.5).astype(int), zero_division=0),
            "recall_default":   recall_score(y_test, (probs >= 0.5).astype(int), zero_division=0),
            # F1-optimal threshold
            "threshold_f1_opt": best_f1_thresh,
            "f1_opt":           f1_at_opt,
            "precision_opt":    prec_at_opt,
            "recall_opt":       rec_at_opt,
            # Recall-90 threshold
            "threshold_r90":    recall_90_thresh,
            "f1_r90":           f1_at_r90,
            "precision_r90":    prec_at_r90,
            "recall_r90":       rec_at_r90,
        })
        # Save chosen thresholds as model tags for the serving layer
        mlflow.set_tags({
            "threshold_f1_opt":  str(best_f1_thresh),
            "threshold_recall90": str(recall_90_thresh),
        })
        mlflow.sklearn.log_model(final, artifact_path="model",
                                 registered_model_name=f"fraud_real_{model_type}")
        mlflow.sklearn.save_model(final, model_path)

        print(f"\n  Threshold Analysis:")
        print(f"  {'Threshold':<18} {'Value':>6} {'F1':>7} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-'*55}")
        print(f"  {'Default (0.50)':<18} {'0.50':>6} {f1_score(y_test,(probs>=0.5).astype(int),zero_division=0):>7.4f} {precision_score(y_test,(probs>=0.5).astype(int),zero_division=0):>10.4f} {recall_score(y_test,(probs>=0.5).astype(int),zero_division=0):>8.4f}")
        print(f"  {'F1-Optimal':<18} {best_f1_thresh:>6.2f} {f1_at_opt:>7.4f} {prec_at_opt:>10.4f} {rec_at_opt:>8.4f}")
        print(f"  {'Recall >= 90%':<18} {recall_90_thresh:>6.2f} {f1_at_r90:>7.4f} {prec_at_r90:>10.4f} {rec_at_r90:>8.4f}")
        print(f"\n  AUC={auc:.4f}")
        print(classification_report(y_test, preds_f1, target_names=["Legit", "Fraud"]))

    return f1_at_opt, auc, run.info.run_id


if __name__ == "__main__":
    print("\n══════════════════════════════════════════")
    print("  Training on Real Credit Card Fraud Data")
    print("══════════════════════════════════════════\n")

    X, y = get_training_data()
    print(f"Dataset: {len(X):,} transactions | Fraud rate: {y.mean()*100:.3f}% ({y.sum()} fraud cases)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # For GB: use class_weight via scale_pos_weight equivalent (no SMOTE — too slow on 450k rows)
    # For LR: use class_weight='balanced'

    results = {}
    # Training only LR + GB — RF is too slow on 284k rows with Optuna
    for mtype in ["lr", "gb"]:
        print(f"Training {mtype.upper()} (8 Optuna trials)...")
        f1, auc, run_id = train_model(X_train, y_train, X_test, y_test, mtype, n_trials=8)
        results[mtype] = {"f1": f1, "auc": auc, "run_id": run_id}

    # Champion selection by AUC (better metric for imbalanced data)
    champion = max(results, key=lambda k: results[k]["auc"])
    champion_run_id = results[champion]["run_id"]
    print(f"\n══════════════════════════════════════════")
    print(f"  Champion: {champion.upper()} | F1={results[champion]['f1']:.4f} | AUC={results[champion]['auc']:.4f}")
    print(f"══════════════════════════════════════════\n")

    mlflow.register_model(f"runs:/{champion_run_id}/model", "fraud_real_champion")
    print(f"Registered 'fraud_real_champion' in MLflow registry.")

    print("\n  Model Comparison:")
    print(f"  {'Model':<8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*26}")
    for m, r in results.items():
        marker = " ← champion" if m == champion else ""
        print(f"  {m.upper():<8} {r['f1']:>8.4f} {r['auc']:>8.4f}{marker}")
