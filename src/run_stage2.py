"""Stage 2: naive baselines, equal-data comparison and trip-diversity test.

Run from the repository root:  python src/run_stage2.py

No TabPFN calls. Every result is appended to results/experiments/experiment_results.csv;
combinations already logged are skipped, so the script can be re-run safely.

Experiments written (feature set A = all_features throughout):

naive_baselines
    GlobalMean, StopMean, RouteStopMean and Ridge on the trip-grouped and
    route-grouped splits (seeds 42-46, full test set, as in the earlier
    experiments) and on the time split (one deterministic split).
equal_data_10k
    XGBoost and LightGBM trained on exactly the 10,000 rows TabPFN's wrapper
    keeps from the trip-grouped training set (seeds 42-46, full test set).
diversity_whole_trips_10k / diversity_random_from_100k_10k
    XGBoost and LightGBM trained on 10,000 rows drawn two ways from the scaling
    experiment's pool: as whole trips, or as random rows from a 100,000-row
    whole-trip sample. Same fixed 20,000-row test set, seeds 0-2. The number of
    distinct trips in each draw is saved to diversity_test_draws.csv.
"""
import numpy as np
import pandas as pd

from data import FEATURE_SETS, REPO_ROOT, load_clean, target
from models import BASELINES, BOOSTERS, make_model, tabpfn_context_idx
from splits import cap_test, get_split, sample_whole_trips
from tracking import EXPERIMENT_RESULTS_CSV, fit_and_score, logged_keys, run_and_log

FEATURE_SET = "all_features"
SEEDS = [42, 43, 44, 45, 46]
DIVERSITY_SEEDS = [0, 1, 2]            # same seeds as the scaling experiment
DIVERSITY_SPLIT = "trip_grouped_test20k"
DIVERSITY_DRAWS_CSV = REPO_ROOT / "results" / "experiments" / "diversity_test_draws.csv"


def key(experiment, split_type, model, seed):
    return {"experiment_name": experiment, "feature_set": FEATURE_SET,
            "split_type": split_type, "model": model, "random_seed": seed}


def pending(experiment, split_type, models, seed, done):
    return [m for m in models if (experiment, FEATURE_SET, split_type, m, seed) not in done]


def check_matches_notebook(df, X, y, cfg):
    """The cleaning here must reproduce a number the notebook already logged."""
    log = pd.read_csv(EXPERIMENT_RESULTS_CSV)
    ref = log[(log.experiment_name == "final_5seed_trip_grouped") & (log.feature_set == FEATURE_SET)
              & (log.model == "LightGBM") & (log.random_seed == 42)]["R2"].iloc[0]
    tr, te = get_split(df, "trip_grouped", 42)
    got = fit_and_score(make_model("LightGBM", cfg, 42), X.iloc[tr], y[tr], X.iloc[te], y[te])["R2"]
    assert abs(got - ref) < 1e-9, f"src/ cleaning does not reproduce the notebook: {got} vs {ref}"
    print(f"Check passed: LightGBM trip-grouped seed 42 R2 = {got:.6f} (logged: {ref:.6f})")


def main():
    df = load_clean()
    cfg = FEATURE_SETS[FEATURE_SET]
    X, y = df[cfg["features"]], target(df)
    print(f"Cleaned dataset: {len(df):,} rows, {len(cfg['features'])} features")
    check_matches_notebook(df, X, y, cfg)
    done = logged_keys()

    # ---- naive baselines ----
    runs = [(s, seed) for s in ["trip_grouped", "route_grouped"] for seed in SEEDS] + [("time", 42)]
    for split_type, seed in runs:
        todo = pending("naive_baselines", split_type, BASELINES, seed, done)
        if not todo:
            continue
        tr, te = get_split(df, split_type, seed)
        if split_type == "time":
            d = df["date"]
            print(f"time split: train {len(tr):,} rows ({d.iloc[tr].min().date()} to {d.iloc[tr].max().date()}), "
                  f"test {len(te):,} rows sampled from {d.iloc[te].min().date()} to {d.iloc[te].max().date()}, "
                  f"routes in test: {df['route_id'].iloc[te].nunique()}")
        for name in todo:
            run_and_log(key("naive_baselines", split_type, name, seed),
                        make_model(name, cfg, seed), X.iloc[tr], y[tr], X.iloc[te], y[te])

    # ---- boosters on TabPFN's 10,000 training rows ----
    for seed in SEEDS:
        todo = pending("equal_data_10k", "trip_grouped", BOOSTERS, seed, done)
        if not todo:
            continue
        tr, te = get_split(df, "trip_grouped", seed)
        sub = tr[tabpfn_context_idx(len(tr), seed)]
        for name in todo:
            run_and_log(key("equal_data_10k", "trip_grouped", name, seed),
                        make_model(name, cfg, seed), X.iloc[sub], y[sub], X.iloc[te], y[te])

    # ---- diversity test: same row count, different number of trips ----
    pool, test = get_split(df, "trip_grouped", 42)       # the scaling experiment's pool and test set
    test = np.random.RandomState(42).choice(test, 20000, replace=False)
    trips = df["trip_id"].values
    draws = []
    for seed in DIVERSITY_SEEDS:
        whole = pool[sample_whole_trips(trips[pool], 10000, np.random.RandomState(seed))]
        whole = whole[tabpfn_context_idx(len(whole), seed)]            # trim to exactly 10,000
        big = pool[sample_whole_trips(trips[pool], 100000, np.random.RandomState(seed))]
        spread = big[tabpfn_context_idx(len(big), seed)]
        for experiment, rows in [("diversity_whole_trips_10k", whole),
                                 ("diversity_random_from_100k_10k", spread)]:
            draws.append({"experiment_name": experiment, "random_seed": seed, "rows": len(rows),
                          "distinct_trips": len(np.unique(trips[rows])),
                          "distinct_dates": df["date"].iloc[rows].nunique()})
            for name in pending(experiment, DIVERSITY_SPLIT, BOOSTERS, seed, done):
                run_and_log(key(experiment, DIVERSITY_SPLIT, name, seed),
                            make_model(name, cfg, seed), X.iloc[rows], y[rows], X.iloc[test], y[test])
    pd.DataFrame(draws).to_csv(DIVERSITY_DRAWS_CSV, index=False)
    print(pd.DataFrame(draws).to_string(index=False))
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
