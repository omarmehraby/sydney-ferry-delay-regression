"""Load and clean the Sydney Ferries dataset.

The cleaning steps are the same as in notebooks/Experiments.ipynb (Step 1), which
in turn copies notebooks/Ferry_Delay_Regression.ipynb. `run_stage2.py` checks that
a model trained on this output reproduces a result already logged by the notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Data-Sydney 1.csv"
CACHE_PATH = REPO_ROOT / "data" / "cache" / "clean.pkl"   # data/ is git-ignored

TARGET = "arrival_delay"
WINSOR_LOW, WINSOR_HIGH = -206, 508
EVENT_KEYS = ["trip_id", "start_date", "current_stop_sequence", "stop_id"]
DROP_COLS = ["Vessels", "vehicle_label", "Precip.", "Wind Gust", "Hour", "Total Capacity", "Ridership"]

CATEGORICAL = ["route_id", "stop_id", "vehicle_id", "Wind", "Condition", "Class"]
NUMERIC = ["start_time", "start_date_ordinal", "month", "day_of_year",
           "current_stop_sequence", "Week", "Dew Point", "Humidity", "Wind Speed",
           "Pressure", "Seating Capacity", "Holidays"]
MISSING_FLAGS = ["stop_id_missing", "Wind_missing", "Condition_missing"]

FEATURE_SETS = {
    "all_features": {
        "categorical": list(CATEGORICAL),
        "numeric": list(NUMERIC),
        "missing": list(MISSING_FLAGS),
    },
}
# Feature set A without four date-derived features, for the time-split follow-up.
# Note: `Week` is the day of the week (0 = Monday), not a week number.
DATE_FEATURES = ["start_date_ordinal", "day_of_year", "month", "Week"]
FEATURE_SETS["no_date_features"] = {
    "categorical": list(CATEGORICAL),
    "numeric": [c for c in NUMERIC if c not in DATE_FEATURES],
    "missing": list(MISSING_FLAGS),
}
for _cfg in FEATURE_SETS.values():
    _cfg["features"] = _cfg["categorical"] + _cfg["numeric"] + _cfg["missing"]


def _extract_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def _start_time_to_minutes(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) * 1440.0
    return pd.to_timedelta(series.astype(str), errors="coerce").dt.total_seconds() / 60.0


def _parse_date(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, origin="1899-12-30", unit="D", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _clean(df_raw):
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df = df_raw.copy()

    for col in ["Dew Point", "Wind Speed", "Pressure", "Humidity"]:
        df[col] = _extract_number(df[col])

    df["start_time"] = _start_time_to_minutes(df["start_time"])
    dt = _parse_date(df["start_date"])
    df["start_date_ordinal"] = (dt - dt.min()).dt.days
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    for col in ["stop_id", "Wind", "Condition"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    weather_num = ["Dew Point", "Humidity", "Wind Speed", "Pressure"]
    agg = {c: ("mean" if c in weather_num else "first") for c in df.columns if c not in EVENT_KEYS}
    df = df.groupby(EVENT_KEYS, dropna=False, as_index=False).agg(agg)

    df[TARGET] = df[TARGET].clip(WINSOR_LOW, WINSOR_HIGH)
    for col in ["Wind", "Condition"]:
        df[col] = df[col].astype("object").fillna("Missing")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["stop_id"] = df["stop_id"].astype("object")
    # Calendar date of each event. Used only to build the time-based split, never as a feature.
    df["date"] = _parse_date(df["start_date"])
    return df


def load_clean(use_cache=True):
    """Return the cleaned, de-duplicated event table (496,042 rows)."""
    stat = DATA_PATH.stat()
    stamp = (stat.st_size, int(stat.st_mtime))
    if use_cache and CACHE_PATH.exists():
        cached = pd.read_pickle(CACHE_PATH)
        if cached.attrs.get("source_stamp") == stamp:
            return cached
    df = _clean(pd.read_csv(DATA_PATH, index_col=0, low_memory=False))
    df.attrs["source_stamp"] = stamp
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE_PATH)
    return df


def target(df):
    return df[TARGET].astype(float).values
