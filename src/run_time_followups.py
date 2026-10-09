"""Two follow-ups on the time split (same fixed 20,000 test rows as Stage 3).

Run from the repository root:  python src/run_time_followups.py [--no-tabpfn]

1. Date features. XGBoost, LightGBM and Ridge without start_date_ordinal,
   day_of_year, month and Week (feature set "no_date_features"), on all training
   rows (experiment "time_split") and on TabPFN's 10,000 rows (experiment
   "equal_data_10k"). Ridge with all features on the 10,000 rows is added so
   every cell has its counterpart. Tests whether the date features hurt when
   predicting later dates. No TabPFN calls.

2. Spread of the TabPFN result. TabPFN on four more random 10,000-row samples
   of the training rows (seeds 43-46; seed 42 was Stage 3), and XGBoost and
   LightGBM on each of the same samples. Feature set A.

Already-logged combinations are skipped. TabPFN runs last and stops cleanly on a
quota error.
"""
import sys

from data import FEATURE_SETS, load_clean, target
from models import BOOSTERS, make_model, tabpfn_context_idx
from splits import get_split
from tracking import QuotaExhausted, logged_keys, run_and_log

SEED = 42
EXTRA_SEEDS = [43, 44, 45, 46]


def main(run_tabpfn=True):
    df = load_clean()
    y = target(df)
    tr, te = get_split(df, "time", SEED)
    sample = {seed: tr[tabpfn_context_idx(len(tr), seed)] for seed in [SEED] + EXTRA_SEEDS}

    # (experiment, feature_set, model, seed, training rows)
    jobs = []
    for name in BOOSTERS + ["Ridge"]:
        jobs.append(("time_split", "no_date_features", name, SEED, tr))
        jobs.append(("equal_data_10k", "no_date_features", name, SEED, sample[SEED]))
    jobs.append(("equal_data_10k", "all_features", "Ridge", SEED, sample[SEED]))
    for seed in EXTRA_SEEDS:
        for name in BOOSTERS:
            jobs.append(("equal_data_10k", "all_features", name, seed, sample[seed]))
    if run_tabpfn:
        for seed in EXTRA_SEEDS:
            jobs.append(("time_split", "all_features", "TabPFN", seed, sample[seed]))

    done = logged_keys()
    jobs = [j for j in jobs if (j[0], j[1], "time", j[2], j[3]) not in done]
    print(f"{len(jobs)} combinations to run.", flush=True)
    while jobs:
        experiment, feature_set, name, seed, rows = jobs[0]
        cfg = FEATURE_SETS[feature_set]
        X = df[cfg["features"]]
        key = {"experiment_name": experiment, "feature_set": feature_set,
               "split_type": "time", "model": name, "random_seed": seed}
        try:
            run_and_log(key, make_model(name, cfg, seed), X.iloc[rows], y[rows], X.iloc[te], y[te])
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
        for experiment, feature_set, name, seed, _ in jobs:
            print("  ", (experiment, feature_set, "time", name, seed))
    else:
        print("Follow-ups complete.")


if __name__ == "__main__":
    main(run_tabpfn="--no-tabpfn" not in sys.argv)
