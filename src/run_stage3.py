"""Stage 3: all models on the time-based split, plus the equal-data comparison.

Run from the repository root:  python src/run_stage3.py [--no-tabpfn]

The time split is deterministic, so there is one run per model (seed 42 sets the
models' own random state and TabPFN's 10,000-row subsample). Every model is
scored on the same fixed sample of 20,000 test rows. The naive baselines for
this split were logged in Stage 2 (experiment "naive_baselines").

Experiments written (feature set A, split_type "time"):

time_split
    XGBoost and LightGBM on all training rows; TabPFN on its 10,000-row subsample.
equal_data_10k
    XGBoost and LightGBM on exactly the 10,000 rows TabPFN receives.

Combinations already logged are skipped. TabPFN runs last; if the hosted API is
out of quota or unavailable the script stops cleanly and lists what is left.
"""
import sys

from data import FEATURE_SETS, load_clean, target
from models import BOOSTERS, make_model, tabpfn_context_idx
from splits import get_split
from tracking import QuotaExhausted, logged_keys, run_and_log

FEATURE_SET = "all_features"
SEED = 42


def main(run_tabpfn=True):
    df = load_clean()
    cfg = FEATURE_SETS[FEATURE_SET]
    X, y = df[cfg["features"]], target(df)
    tr, te = get_split(df, "time", SEED)
    sub = tr[tabpfn_context_idx(len(tr), SEED)]
    print(f"time split: {len(tr):,} training rows, {len(te):,} test rows; "
          f"TabPFN subsample {len(sub):,} rows covering "
          f"{df['trip_id'].iloc[sub].nunique():,} trips and {df['route_id'].iloc[sub].nunique()} routes")

    jobs = [("time_split", name, tr) for name in BOOSTERS]
    jobs += [("equal_data_10k", name, sub) for name in BOOSTERS]
    if run_tabpfn:
        jobs.append(("time_split", "TabPFN", tr))      # the wrapper subsamples to `sub`

    done = logged_keys()
    jobs = [j for j in jobs if (j[0], FEATURE_SET, "time", j[1], SEED) not in done]
    print(f"{len(jobs)} combinations to run.")
    while jobs:
        experiment, name, rows = jobs[0]
        key = {"experiment_name": experiment, "feature_set": FEATURE_SET,
               "split_type": "time", "model": name, "random_seed": SEED}
        try:
            run_and_log(key, make_model(name, cfg, SEED), X.iloc[rows], y[rows], X.iloc[te], y[te])
        except QuotaExhausted as exc:
            print(f"\nSTOPPED: TabPFN quota exhausted ({exc}).")
            break
        except Exception as exc:
            if name != "TabPFN":
                raise
            print(f"\nSTOPPED: TabPFN call failed ({type(exc).__name__}: {str(exc)[:300]}).")
            break
        jobs.pop(0)
    if jobs:
        print(f"{len(jobs)} combinations still missing:")
        for experiment, name, _ in jobs:
            print("  ", (experiment, FEATURE_SET, "time", name, SEED))
    else:
        print("Stage 3 complete.")


if __name__ == "__main__":
    main(run_tabpfn="--no-tabpfn" not in sys.argv)
