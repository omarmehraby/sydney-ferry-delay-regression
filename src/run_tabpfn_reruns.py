"""TabPFN re-runs on the currently served model, and TabPFN with more training rows.

Run from the repository root:  python src/run_tabpfn_reruns.py [--no-tabpfn]

Part 1 -- re-run the August TabPFN results on today's model (feature set A,
TabPFN restricted to the same random 10,000 training rows and scored on the same
full test sets as in August), in this order:

    a. trip-grouped split, seeds 42-46
    b. unseen-route (route-grouped) split, the same five splits (seeds 42-46)
    c. random split, seed 42

These are logged under experiment "tabpfn_rerun". The August rows stay in the
log under their existing keys.

Part 2 -- time split, same fixed 20,000 test rows as Stage 3:

    TabPFN on a random 50,000-row sample   experiment "time_split_tabpfn_50k"
    TabPFN on all training rows            experiment "time_split_tabpfn_all"
    XGBoost and LightGBM on the same 50,000 rows   experiment "equal_data_50k"

Part 3 -- trip-grouped split scored on a fixed 20,000-row sample of the test set
(split_type "trip_grouped_fixed20k", seeds 42 and 43):

    XGBoost, LightGBM on all rows; TabPFN on 10,000 rows   experiment "trip_fixed20k"
    XGBoost, LightGBM on TabPFN's 10,000 rows               experiment "trip_fixed20k_equal10k"
    TabPFN on all training rows                             experiment "trip_fixed20k_tabpfn_all"

and two more random 50,000-row samples on the time split (seeds 43, 44) for
TabPFN, XGBoost and LightGBM.

Part 4 -- date-grouped split (split_type "date_grouped_fixed20k", seeds 42 and
43): 20% of calendar dates held out at random, scored on a fixed 20,000-row
sample of the test set. Baselines, XGBoost and LightGBM on all rows and TabPFN
on 10,000 rows (experiment "date_grouped"), the boosters on TabPFN's 10,000
rows ("date_grouped_equal10k") and TabPFN on all rows
("date_grouped_tabpfn_all"). It tests whether results on the trip-grouped
split depend on calendar dates being shared between training and test.

For every TabPFN row, the model version the service reports as its default at
the time of the call is written to results/experiments/tabpfn_run_log.csv.

Resumable: logged combinations are skipped. Connection failures are retried;
on a quota error the script stops and lists what is left.
"""
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from data import FEATURE_SETS, REPO_ROOT, load_clean, target
from models import BASELINES, BOOSTERS, OrdinalEncoded, make_model, tabpfn_context_idx
from splits import cap_test, get_split
from tracking import KEY_COLUMNS, QuotaExhausted, fit_and_score, log_result, logged_keys

FEATURE_SET = "all_features"
SEEDS = [42, 43, 44, 45, 46]
RETRY_WAITS = [30, 60, 120, 300, 600]
RUN_LOG = REPO_ROOT / "results" / "experiments" / "tabpfn_run_log.csv"
RUN_LOG_COLUMNS = KEY_COLUMNS + ["served_default_model_version", "tabpfn_client_version",
                                 "run_at_utc", "n_train_rows", "n_test_rows"]


def served_model_version():
    """The default model version the hosted service reports right now."""
    import tabpfn_client
    from tabpfn_client.client import ServiceClient
    tabpfn_client.init()
    response = ServiceClient.httpx_client.get("/tabpfn/get_settings")
    response.raise_for_status()
    return response.json()["default_model_version"]


def uncapped_tabpfn(cfg):
    """TabPFN on exactly the rows it is given (no 10,000-row restriction)."""
    from tabpfn_client import TabPFNRegressor
    return OrdinalEncoded(cfg["categorical"], TabPFNRegressor())


TRIP_FIXED = "trip_grouped_fixed20k"


def trip_fixed_jobs(seed):
    """All models on the trip-grouped split, scored on the same 20,000 test rows."""
    return ([("trip_fixed20k", TRIP_FIXED, m, seed, "all") for m in BOOSTERS]
            + [("trip_fixed20k_equal10k", TRIP_FIXED, m, seed, "capped") for m in BOOSTERS]
            + [("trip_fixed20k", TRIP_FIXED, "TabPFN", seed, "capped"),
               ("trip_fixed20k_tabpfn_all", TRIP_FIXED, "TabPFN", seed, "all")])


DATE_GROUPED = "date_grouped_fixed20k"


def date_grouped_jobs(seed):
    """Every model on the date-grouped split; quota-free models first."""
    return ([("date_grouped", DATE_GROUPED, m, seed, "all") for m in BASELINES + BOOSTERS]
            + [("date_grouped_equal10k", DATE_GROUPED, m, seed, "capped") for m in BOOSTERS]
            + [("date_grouped", DATE_GROUPED, "TabPFN", seed, "capped"),
               ("date_grouped_tabpfn_all", DATE_GROUPED, "TabPFN", seed, "all")])


def main():
    df = load_clean()
    cfg = FEATURE_SETS[FEATURE_SET]
    X, y = df[cfg["features"]], target(df)

    # (experiment, split_type, model, seed, how to pick training rows)
    jobs = [("tabpfn_rerun", "trip_grouped", "TabPFN", s, "capped") for s in SEEDS]
    jobs += [("tabpfn_rerun", "route_grouped", "TabPFN", s, "capped") for s in SEEDS]
    jobs += [("tabpfn_rerun", "random", "TabPFN", 42, "capped")]
    jobs += [("equal_data_50k", "time", m, 42, "sample50k") for m in BOOSTERS]
    jobs += [("time_split_tabpfn_50k", "time", "TabPFN", 42, "sample50k"),
             ("time_split_tabpfn_all", "time", "TabPFN", 42, "all")]
    # Part 3: trip-grouped split scored on a fixed 20,000-row sample of the test set
    # (seed 42 first; seed 43 at the very end), and two more 50,000-row samples on
    # the time split so that row has a spread.
    jobs += trip_fixed_jobs(42)
    for seed in (43, 44):
        jobs += [("equal_data_50k", "time", m, seed, "sample50k") for m in BOOSTERS]
        jobs += [("time_split_tabpfn_50k", "time", "TabPFN", seed, "sample50k")]
    jobs += trip_fixed_jobs(43)
    # Part 4: date-grouped split (seed 42, then 43).
    jobs += date_grouped_jobs(42) + date_grouped_jobs(43)

    if "--no-tabpfn" in sys.argv:          # run only what needs no quota
        jobs = [j for j in jobs if j[2] != "TabPFN"]
    done = logged_keys()
    jobs = [j for j in jobs if (j[0], FEATURE_SET, j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} combinations to run.", flush=True)
    failures, stopped = 0, None
    while jobs:
        experiment, split_type, name, seed, rows_rule = jobs[0]
        if split_type == TRIP_FIXED:
            tr, te = get_split(df, "trip_grouped", seed)
            te = cap_test(te)                             # fixed 20,000-row sample of the test set
        else:
            tr, te = get_split(df, split_type, seed)
        if rows_rule == "sample50k":
            tr = tr[np.random.RandomState(seed).choice(len(tr), 50000, replace=False)]
        if name != "TabPFN" and rows_rule == "capped":
            tr = tr[tabpfn_context_idx(len(tr), seed)]    # the rows TabPFN's 10,000-row sample uses
        if name != "TabPFN":
            model = make_model(name, cfg, seed)
        elif rows_rule == "capped":
            model = make_model("TabPFN", cfg, seed)      # same 10,000-row sample as in August
        else:
            model = uncapped_tabpfn(cfg)
        key = {"experiment_name": experiment, "feature_set": FEATURE_SET,
               "split_type": split_type, "model": name, "random_seed": seed}
        try:
            served = served_model_version() if name == "TabPFN" else None
            res = fit_and_score(model, X.iloc[tr], y[tr], X.iloc[te], y[te])
            failures = 0
        except QuotaExhausted as exc:
            stopped = f"TabPFN quota exhausted ({exc})"
            break
        except Exception as exc:
            if name != "TabPFN":
                raise
            failures += 1
            if failures <= len(RETRY_WAITS):
                wait = RETRY_WAITS[failures - 1]
                print(f"TabPFN call failed ({type(exc).__name__}: {str(exc)[:200]}); "
                      f"retry {failures}/{len(RETRY_WAITS)} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            stopped = f"TabPFN call failed ({type(exc).__name__}: {str(exc)[:400]})"
            break
        log_result({**key, **res})
        if name == "TabPFN":
            n_train = min(len(tr), 10000) if rows_rule == "capped" else len(tr)
            log_result({**key, "served_default_model_version": served,
                        "tabpfn_client_version": version("tabpfn-client"),
                        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "n_train_rows": n_train, "n_test_rows": len(te)},
                       path=RUN_LOG, columns=RUN_LOG_COLUMNS)
        print(f"[{experiment}] {split_type:13s} {name:8s} seed={seed} train={len(tr):,} test={len(te):,} "
              f"R2={res['R2']:.4f} RMSE={res['RMSE']:.2f} fit={res['training_time']:.0f}s "
              f"predict={res['prediction_time']:.0f}s" + (f" model={served}" if served else ""), flush=True)
        jobs.pop(0)

    if stopped:
        print(f"\nSTOPPED: {stopped}")
    if jobs:
        print(f"{len(jobs)} combinations still missing:")
        for j in jobs:
            print("  ", j[:4])
    else:
        print("All re-runs complete.")


if __name__ == "__main__":
    main()
