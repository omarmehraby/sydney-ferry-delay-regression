"""Time-respecting few-shot route experiment (robustness check for Stage 4).

Run from the repository root:  python src/run_stage4_time.py [--no-tabpfn]

Same k values, conditions and models as src/run_stage4.py, but nothing in
training is later than anything in testing:

- The cutoff is the time split's boundary (training dates up to 12 May 2023,
  test dates from 13 May to 31 July 2023; later rows are dropped).
- For each route, the adaptation rows are the k most recent rows of that route
  before the cutoff, taken sailing by sailing from the latest backwards. A
  sailing is one trip_id on one date (a trip_id is a scheduled service that can
  recur on several dates). The oldest sailing included is cut to its last stops
  so that exactly k rows are used. n_target_trips counts sailings here.
- The test set is a fixed sample of at most 2,000 rows of that route from the
  test period.
- Rows of the other eight routes come only from before the cutoff.

The adaptation rows are fixed by the calendar, so there is one run per setting
(seed 0, which sets the models' random state and TabPFN's sample of other-route
rows). Results go to results/experiments/fewshot_time_results.csv and, as
experiment "fewshot_time", to experiment_results.csv. Resumable; TabPFN runs
last and stops cleanly on a quota error.
"""
import sys

import numpy as np
import pandas as pd

from data import REPO_ROOT, load_clean
from run_stage4 import SPLIT_SEED, TEST_CAP, Sets, run
from splits import cap_test, time_split

OUT_CSV = REPO_ROOT / "results" / "experiments" / "fewshot_time_results.csv"
SEEDS = [0]


class TimeSets(Sets):
    """Row positions for one new route under the time cutoff."""
    def __init__(self, df, route, before, test_period):
        is_route = (df["route_id"] == route).values
        self.test = cap_test(np.where(is_route & test_period)[0], cap=TEST_CAP, seed=SPLIT_SEED)
        self.others = np.where(~is_route & before)[0]
        # the route's rows before the cutoff, most recent first; a trip's rows stay together
        pool = np.where(is_route & before)[0]
        order = np.lexsort((-df["current_stop_sequence"].values[pool],
                            pd.factorize(df["trip_id"].values[pool])[0],
                            -df["start_time"].values[pool],
                            -df["date"].values[pool].astype("int64")))
        self.pool = pool[order]
        # A trip_id is a scheduled service that can recur on several dates, so a
        # single sailing is identified here by (trip_id, date).
        self.pool_sailings = list(zip(df["trip_id"].values[self.pool], df["date"].values[self.pool]))

    def target_rows(self, k, seed):
        rows = self.pool[:k]
        return np.sort(rows), len(set(self.pool_sailings[:k]))


def main(run_tabpfn=True):
    df = load_clean()
    train_idx, _ = time_split(df)
    date = df["date"].values
    last_train = date[train_idx].max()
    before = date <= last_train
    kept = np.zeros(len(df), dtype=bool)
    kept[train_idx] = True
    assert (kept == before).all()                      # same training period as Stage 3
    test_period = (date > last_train) & (date <= np.datetime64("2023-07-31"))
    routes = sorted(df["route_id"].unique())
    sets = {r: TimeSets(df, r, before, test_period) for r in routes}
    for r in routes:
        s = sets[r]
        newest = s.pool[:5000]
        print(f"{r}: {len(s.pool):,} rows before the cutoff, test {len(s.test):,} rows, "
              f"other routes {len(s.others):,} rows; the 5,000 most recent rows span "
              f"{str(date[newest].min())[:10]} to {str(date[newest].max())[:10]}")
    run(df, sets, SEEDS, OUT_CSV, "fewshot_time", run_tabpfn)


if __name__ == "__main__":
    main(run_tabpfn="--no-tabpfn" not in sys.argv)
