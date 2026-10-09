"""Stage 4: few-shot route experiment.

Run from the repository root:  python src/run_stage4.py [--no-tabpfn]

Each of the nine routes in turn is treated as a "new" route:

- Its trips are split (grouped by trip_id, 80/20, seed 42) into an adaptation
  pool and a test set. The test set is a fixed sample of at most 2,000 rows,
  identical for every model, k and seed.
- For k in {0, 100, 500, 1000, 5000} and seeds 0-2, k rows are drawn from the
  adaptation pool trip by trip (see splits.sample_trips_exact).
- Conditions:
    pooled        all rows of the other eight routes plus the k rows
    target_only   the k rows alone (k > 0)
    pooled_10k    XGBoost and LightGBM on exactly the rows TabPFN receives in
                  the pooled condition (equal-data comparison)
- TabPFN, pooled: the training set always exceeds 10,000 rows, so it receives
  all k target-route rows plus a random sample of other-route rows, 10,000 in
  total.
- Models: XGBoost, LightGBM, TabPFN and the four naive baselines, feature set A,
  default settings.

Results go to results/experiments/fewshot_results.csv (one row per route,
condition, k, model and seed) and, in the usual format, to experiment_results.csv.
Finished combinations are skipped on re-run. Everything that needs no TabPFN
quota runs first; TabPFN then runs seed by seed, so each finished seed is a
complete result. On a quota error the script stops and lists what is left.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from data import FEATURE_SETS, REPO_ROOT, load_clean, target
from models import BASELINES, BOOSTERS, TABPFN_MAX_CONTEXT, make_model
from splits import cap_test, sample_trips_exact
from tracking import QuotaExhausted, fit_and_score, log_result, logged_keys

FEATURE_SET = "all_features"
K_VALUES = [0, 100, 500, 1000, 5000]
SEEDS = [0, 1, 2]
TEST_CAP = 2000
SPLIT_SEED = 42
RETRY_WAITS = [30, 60, 120, 300, 600]   # seconds, for TabPFN connection failures
FEWSHOT_CSV = REPO_ROOT / "results" / "experiments" / "fewshot_results.csv"
FEWSHOT_KEY = ["route", "condition", "k", "model", "random_seed"]
FEWSHOT_COLUMNS = FEWSHOT_KEY + ["n_target_rows", "n_target_trips", "n_train_rows", "n_test_rows",
                                 "MAE", "RMSE", "R2", "training_time", "prediction_time"]
MAIN_KEY = ["experiment_name", "feature_set", "split_type", "model", "random_seed"]


def fewshot_done(path=FEWSHOT_CSV):
    if not path.is_file():
        return set()
    done = pd.read_csv(path, usecols=FEWSHOT_KEY)
    return set(done[FEWSHOT_KEY].itertuples(index=False, name=None))


class Sets:
    """Row positions for one new route: test set, adaptation pool, other routes."""
    def __init__(self, df, route):
        is_route = (df["route_id"] == route).values
        route_rows = np.where(is_route)[0]
        pool, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
                          .split(route_rows, groups=df["trip_id"].values[route_rows]))
        self.pool = route_rows[pool]
        self.test = cap_test(route_rows[test], cap=TEST_CAP, seed=SPLIT_SEED)
        self.others = np.where(~is_route)[0]
        self.pool_trips = df["trip_id"].values[self.pool]

    def target_rows(self, k, seed):
        """(rows, distinct trips) of the k adaptation rows for this seed."""
        idx, n_trips = sample_trips_exact(self.pool_trips, k, np.random.RandomState(seed))
        return self.pool[idx], n_trips

    def tabpfn_pooled_rows(self, target_rows, seed):
        """All target rows plus random other-route rows, 10,000 in total."""
        fill = np.random.RandomState(10_000 + seed).choice(
            self.others, TABPFN_MAX_CONTEXT - len(target_rows), replace=False)
        return np.concatenate([target_rows, fill])

    def train_rows(self, condition, model, target_rows, seed):
        if condition == "target_only":
            return target_rows
        if condition == "pooled_10k" or model == "TabPFN":
            return self.tabpfn_pooled_rows(target_rows, seed)
        return np.concatenate([self.others, target_rows])


def all_jobs(routes, run_tabpfn, seeds=SEEDS):
    """Quota-free jobs first; then TabPFN seed by seed."""
    free, tabpfn = [], []
    for seed in seeds:
        for route in routes:
            for k in K_VALUES:
                for model in BOOSTERS + BASELINES:
                    free.append((route, "pooled", k, model, seed))
                    if k > 0:
                        free.append((route, "target_only", k, model, seed))
                for model in BOOSTERS:
                    free.append((route, "pooled_10k", k, model, seed))
                tabpfn.append((route, "pooled", k, "TabPFN", seed))
                if k > 0:
                    tabpfn.append((route, "target_only", k, "TabPFN", seed))
    return free + (tabpfn if run_tabpfn else [])


def main(run_tabpfn=True):
    df = load_clean()
    routes = sorted(df["route_id"].unique())
    sets = {r: Sets(df, r) for r in routes}
    for r in routes:
        print(f"{r}: adaptation pool {len(sets[r].pool):,} rows, test {len(sets[r].test):,} rows, "
              f"other routes {len(sets[r].others):,} rows")
    run(df, sets, SEEDS, FEWSHOT_CSV, "fewshot", run_tabpfn)


def run(df, sets, seeds, out_csv, experiment, run_tabpfn):
    """Run every pending combination for the route sets in `sets` and log it to
    `out_csv` and, under experiment name `experiment`, to experiment_results.csv."""
    cfg = FEATURE_SETS[FEATURE_SET]
    X, y = df[cfg["features"]], target(df)
    routes = sorted(sets)
    done = fewshot_done(out_csv)
    jobs = [j for j in all_jobs(routes, run_tabpfn, seeds) if j not in done]
    total = len(jobs)
    print(f"{total} combinations to run ({sum(j[3] == 'TabPFN' for j in jobs)} TabPFN).", flush=True)
    in_main_log = logged_keys()
    cache = {}
    stopped = None
    failures = 0
    while jobs:
        route, condition, k, model, seed = jobs[0]
        s = sets[route]
        if (route, k, seed) not in cache:
            cache = {(route, k, seed): s.target_rows(k, seed)}
        target_rows, n_trips = cache[(route, k, seed)]
        train = s.train_rows(condition, model, target_rows, seed)
        try:
            res = fit_and_score(make_model(model, cfg, seed), X.iloc[train], y[train],
                                X.iloc[s.test], y[s.test])
            failures = 0
        except QuotaExhausted as exc:
            stopped = f"TabPFN quota exhausted ({exc})"
            break
        except Exception as exc:
            if model != "TabPFN":
                raise
            # connection drops are retried; nothing is logged for a failed call
            failures += 1
            if failures <= len(RETRY_WAITS):
                wait = RETRY_WAITS[failures - 1]
                print(f"TabPFN call failed ({type(exc).__name__}: {str(exc)[:120]}); "
                      f"retry {failures}/{len(RETRY_WAITS)} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            stopped = f"TabPFN call failed ({type(exc).__name__}: {str(exc)[:400]})"
            break
        main_key = (experiment, FEATURE_SET, f"{experiment}_{route}_{condition}_k{k}", model, seed)
        if main_key not in in_main_log:
            log_result({**dict(zip(MAIN_KEY, main_key)), **res})
        log_result({"route": route, "condition": condition, "k": k, "model": model,
                    "random_seed": seed, "n_target_rows": len(target_rows),
                    "n_target_trips": n_trips, "n_train_rows": len(train),
                    "n_test_rows": len(s.test), **res}, path=out_csv, columns=FEWSHOT_COLUMNS)
        jobs.pop(0)
        n_done = total - len(jobs)
        if model == "TabPFN" or n_done % 50 == 0:
            print(f"[{n_done}/{total}] {route} {condition:11s} k={k:<5d} {model:13s} seed={seed} "
                  f"R2={res['R2']:7.3f} RMSE={res['RMSE']:6.1f}", flush=True)

    if stopped:
        print(f"\nSTOPPED: {stopped}")
    if jobs:
        left = pd.DataFrame(jobs, columns=FEWSHOT_KEY)
        print(f"{len(jobs)} combinations still missing, by model and seed:")
        print(left.groupby(["model", "random_seed"]).size().to_string())
        print("next:", jobs[0])
    else:
        print(f"{experiment}: complete.")


if __name__ == "__main__":
    main(run_tabpfn="--no-tabpfn" not in sys.argv)
