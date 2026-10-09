"""Models and naive baselines. Every estimator takes a DataFrame of raw features.

XGBoost, LightGBM and TabPFN use the same untuned settings and ordinal encoding as
the notebooks. No hyperparameters are tuned anywhere.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

TABPFN_MAX_CONTEXT = 10000

BOOSTERS = ["XGBoost", "LightGBM"]
BASELINES = ["GlobalMean", "StopMean", "RouteStopMean", "Ridge"]


class OrdinalEncoded:
    """Ordinal-encode the categoricals, then fit `model` (the notebooks' procedure)."""
    def __init__(self, cat_cols, model):
        self.model = model
        self.encoder = ColumnTransformer(
            [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
            remainder="passthrough")

    def fit(self, X, y):
        self.model.fit(self.encoder.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.encoder.transform(X))


def tabpfn_context_idx(n_rows, seed):
    """Row positions TabPFN's wrapper keeps when given `n_rows` training rows."""
    if n_rows <= TABPFN_MAX_CONTEXT:
        return np.arange(n_rows)
    return np.random.RandomState(seed).choice(n_rows, TABPFN_MAX_CONTEXT, replace=False)


class CapAtContext:
    """Restrict TabPFN to a random 10,000-row sample of the training rows.

    This is the project's own restriction, not a limit of the hosted service."""
    def __init__(self, est, seed):
        self.est, self.seed, self.was_capped = est, seed, False

    def fit(self, X, y):
        if len(X) > TABPFN_MAX_CONTEXT:
            self.was_capped = True
            idx = tabpfn_context_idx(len(X), self.seed)
            X, y = X[idx], y[idx]
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)


class MeanBaseline:
    """Predict the training-set mean delay of a key, falling back to coarser keys.

    `levels` is a list of column lists, most specific first. A test row whose key
    was not seen in training at one level falls back to the next level, and
    finally to the global training mean. All statistics come from training rows.
    """
    def __init__(self, levels):
        self.levels = levels

    def fit(self, X, y):
        self.global_mean_ = float(np.mean(y))
        frame = X.assign(_y=y)
        self.tables_ = [frame.groupby(cols)["_y"].mean().rename("_pred").reset_index()
                        for cols in self.levels]
        return self

    def predict(self, X):
        pred = pd.Series(np.nan, index=X.index, dtype=float)
        for cols, table in zip(self.levels, self.tables_):
            merged = X[cols].merge(table, on=cols, how="left")["_pred"].values
            pred = pred.where(pred.notna(), merged)
        return pred.fillna(self.global_mean_).values


def _as_str(frame):
    # missing IDs become their own category
    return frame.astype(str)


def make_ridge(cat_cols, num_cols):
    """Ridge regression (default alpha) on one-hot categoricals and standardised numerics."""
    pre = ColumnTransformer([
        ("cat", make_pipeline(FunctionTransformer(_as_str), OneHotEncoder(handle_unknown="ignore")), cat_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge())])


def make_model(name, cfg, seed):
    """Build one estimator by name. `cfg` is an entry of data.FEATURE_SETS."""
    cat, other = cfg["categorical"], cfg["numeric"] + cfg["missing"]
    if name == "XGBoost":
        return OrdinalEncoded(cat, XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, n_jobs=-1, tree_method="hist"))
    if name == "LightGBM":
        return OrdinalEncoded(cat, LGBMRegressor(
            n_estimators=300, learning_rate=0.1, num_leaves=31, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1))
    if name == "TabPFN":
        from tabpfn_client import TabPFNRegressor      # hosted API
        return OrdinalEncoded(cat, CapAtContext(TabPFNRegressor(), seed))
    if name == "GlobalMean":
        return MeanBaseline([])
    if name == "StopMean":
        return MeanBaseline([["stop_id"]])
    if name == "RouteStopMean":
        return MeanBaseline([["route_id", "stop_id"], ["stop_id"]])
    if name == "Ridge":
        return make_ridge(cat, other)
    raise ValueError(f"Unknown model: {name!r}")
