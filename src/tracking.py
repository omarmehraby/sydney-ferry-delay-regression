"""Append-only experiment log, with the same columns as notebooks/Experiments.ipynb."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import REPO_ROOT

EXPERIMENT_RESULTS_CSV = REPO_ROOT / "results" / "experiments" / "experiment_results.csv"
RESULT_COLUMNS = ["experiment_name", "feature_set", "split_type", "model",
                  "MAE", "RMSE", "R2", "training_time", "prediction_time", "random_seed"]
KEY_COLUMNS = ["experiment_name", "feature_set", "split_type", "model", "random_seed"]


class QuotaExhausted(Exception):
    pass


def score(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def logged_keys(path=EXPERIMENT_RESULTS_CSV):
    """Set of (experiment, feature_set, split_type, model, seed) already on disk."""
    if not path.is_file():
        return set()
    done = pd.read_csv(path, usecols=KEY_COLUMNS)
    return set(done[KEY_COLUMNS].itertuples(index=False, name=None))


def log_result(row, path=EXPERIMENT_RESULTS_CSV, columns=RESULT_COLUMNS):
    """Append one row. Never rewrites anything already in the file."""
    pd.DataFrame([{c: row.get(c) for c in columns}], columns=columns).to_csv(
        path, mode="a", header=not path.is_file(), index=False)


def is_quota_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return (getattr(exc, "status_code", None) == 429 or "429" in text
            or "quota" in text or "usage limit" in text or "too many requests" in text)


def fit_and_score(model, X_train, y_train, X_test, y_test):
    """Train, predict and score one model. Raises QuotaExhausted on a TabPFN 429."""
    try:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_t = time.perf_counter() - t0
    except Exception as exc:
        if is_quota_error(exc):
            raise QuotaExhausted(str(exc)) from exc
        raise
    return {**score(y_test, y_pred), "training_time": train_t, "prediction_time": pred_t}


def run_and_log(key, model, X_train, y_train, X_test, y_test):
    """Run one combination and append it to the log. `key` is a dict of KEY_COLUMNS."""
    row = {**key, **fit_and_score(model, X_train, y_train, X_test, y_test)}
    log_result(row)
    print(f"[{key['experiment_name']}] {key['split_type']:20s} {key['model']:14s} "
          f"seed={key['random_seed']}  RMSE={row['RMSE']:7.2f}  MAE={row['MAE']:6.2f}  "
          f"R2={row['R2']:7.4f}  train={row['training_time']:6.2f}s  "
          f"predict={row['prediction_time']:5.2f}s", flush=True)
    return row
