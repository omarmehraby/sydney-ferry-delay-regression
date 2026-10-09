"""Compute the dataset facts quoted in the paper.

Run from the repository root:  python src/make_dataset_summary.py

Needs the raw CSV. Writes

    results/dataset/dataset_summary.csv   one (key, value) row per fact
    results/dataset/dataset_routes.csv    events, trips and stops per route

so that the paper's numbers can be regenerated without the data file.
"""
import numpy as np
import pandas as pd

from data import DATA_PATH, EVENT_KEYS, REPO_ROOT, TARGET, WINSOR_HIGH, WINSOR_LOW, load_clean
from splits import FIXED_TEST_CAP, TIME_SPLIT_LAST_DATE, time_split

OUT = REPO_ROOT / "results" / "dataset"


def main():
    raw = pd.read_csv(DATA_PATH, index_col=0, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    df = load_clean()
    facts = {}

    facts["raw_rows"] = len(raw)
    facts["raw_columns"] = raw.shape[1]
    copies = raw.groupby(EVENT_KEYS, dropna=False).size()
    facts["events"] = len(copies)
    facts["multi_copy_events"] = int((copies > 1).sum())
    facts["max_copies_per_event"] = int(copies.max())
    delay_values = raw.groupby(EVENT_KEYS, dropna=False)[TARGET].nunique()
    facts["delay_conflict_events"] = int((delay_values > 1).sum())

    facts["raw_delay_min"] = float(raw[TARGET].min())
    facts["raw_delay_max"] = float(raw[TARGET].max())
    facts["raw_delay_p01"] = float(raw[TARGET].quantile(0.01))
    facts["raw_delay_p99"] = float(raw[TARGET].quantile(0.99))
    facts["winsor_low"] = WINSOR_LOW
    facts["winsor_high"] = WINSOR_HIGH
    facts["target_mean"] = float(df[TARGET].mean())
    facts["target_std"] = float(df[TARGET].std())
    facts["events_clipped_low"] = int((raw.drop_duplicates(EVENT_KEYS)[TARGET] < WINSOR_LOW).sum())
    facts["events_clipped_high"] = int((raw.drop_duplicates(EVENT_KEYS)[TARGET] > WINSOR_HIGH).sum())

    date = df["date"]
    facts["date_min"] = str(date.min().date())
    facts["date_max"] = str(date.max().date())
    facts["distinct_dates"] = int(date.nunique())
    months = pd.period_range(date.min(), date.max(), freq="M")
    present = set(date.dt.to_period("M").unique())
    facts["months_without_rows"] = " ".join(str(m) for m in months if m not in present)
    facts["n_routes"] = int(df["route_id"].nunique())
    facts["n_trips"] = int(df["trip_id"].nunique())
    facts["n_stops"] = int(df["stop_id"].nunique())
    facts["n_vehicles"] = int(df["vehicle_id"].nunique())
    facts["events_missing_stop_id"] = int(df["stop_id"].isna().sum())

    after = date > TIME_SPLIT_LAST_DATE
    facts["time_split_last_date"] = TIME_SPLIT_LAST_DATE
    facts["events_after_cutoff"] = int(after.sum())
    facts["routes_after_cutoff"] = " ".join(sorted(df.loc[after, "route_id"].unique()))
    tr, te = time_split(df)
    period = (~after) & (date > date.iloc[tr].max())
    facts["time_train_rows"] = len(tr)
    facts["time_train_dates"] = int(date.iloc[tr].nunique())
    facts["time_train_first"] = str(date.iloc[tr].min().date())
    facts["time_train_last"] = str(date.iloc[tr].max().date())
    facts["time_test_period_rows"] = int(period.sum())
    facts["time_test_dates"] = int(date[period].nunique())
    facts["time_test_first"] = str(date[period].min().date())
    facts["time_test_last"] = str(date[period].max().date())
    facts["time_test_rows"] = len(te)
    facts["time_test_routes"] = int(df["route_id"].iloc[te].nunique())
    facts["fixed_test_cap"] = FIXED_TEST_CAP

    routes = df.groupby("route_id").agg(
        events=("trip_id", "size"), trips=("trip_id", "nunique"), stops=("stop_id", "nunique"),
        first_date=("date", "min"), last_date=("date", "max"),
        mean_delay=(TARGET, "mean"), std_delay=(TARGET, "std")).reset_index()
    routes["share_pct"] = 100 * routes["events"] / routes["events"].sum()
    routes["rows_per_trip"] = routes["events"] / routes["trips"]
    for c in ["first_date", "last_date"]:
        routes[c] = routes[c].dt.date

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(facts.items()), columns=["key", "value"]).to_csv(OUT / "dataset_summary.csv", index=False)
    routes.round(3).to_csv(OUT / "dataset_routes.csv", index=False)
    print(pd.DataFrame(list(facts.items()), columns=["key", "value"]).to_string(index=False))
    print(routes.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
