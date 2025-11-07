# scripts/train_model.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_PATH = "data/health_data.csv"
MODEL_PATH = "models/anomaly_model.pkl"
CONFIG_PATH = "models/config.json"

def find_best_threshold(y_true, y_proba):
    # Search thresholds for best F1
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_hat)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def train():
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["heart_rate", "sleep_hours", "activity_level", "temperature"]
    X = df[feature_cols].values
    y = df["anomaly"].values

    # Hold-out test split first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5-fold CV to tune simple RF hyperparams & threshold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_cfg = None
    best_cv_f1 = -1.0
    best_threshold = 0.5

    # A few robust configs (kept small for speed)
    grid = [
        dict(n_estimators=300, max_depth=None, min_samples_leaf=2),
        dict(n_estimators=400, max_depth=12, min_samples_leaf=1),
        dict(n_estimators=500, max_depth=None, min_samples_leaf=1),
    ]

    for cfg in grid:
        f1s = []
        ths = []
        for tr_idx, va_idx in skf.split(X_trainval, y_trainval):
            X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
            y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                **cfg
            )
            model.fit(X_tr, y_tr)
            proba_va = model.predict_proba(X_va)[:, 1]
            th, f1 = find_best_threshold(y_va, proba_va)
            f1s.append(f1)
            ths.append(th)

        mean_f1 = float(np.mean(f1s))
        mean_th = float(np.mean(ths))
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_cfg = cfg
            best_threshold = mean_th

    # Train final on all train+val with best params
    final_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **best_cfg
    )
    final_model.fit(X_trainval, y_trainval)

    # Evaluate on test
    test_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_proba)

    print("==== Test Metrics ====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"Chosen threshold: {best_threshold:.3f} (from CV best F1={best_cv_f1:.4f})")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    config = {
        "feature_order": feature_cols,
        "threshold": best_threshold,
        # Safety-net rules (can tweak here)
        "rule_safety_net": {
            "hr_high": 110,
            "sleep_low": 5.0,
            "temp_high": 37.8
        }
    }
    Path(CONFIG_PATH).write_text(json.dumps(config, indent=2))
    print(f"Saved config → {CONFIG_PATH}")

if __name__ == "__main__":
    train()
