"""Train/test splits. Each function returns (train_idx, test_idx) as row positions in `df`."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TIME_SPLIT_LAST_DATE = "2023-07-31"   # later rows are F3-only (source file cut off mid-August 2023)
TIME_SPLIT_TRAIN_FRAC = 0.80
FIXED_TEST_CAP = 20000                # same fixed test sample for every model
FIXED_TEST_SEED = 42


def cap_test(test_idx, cap=FIXED_TEST_CAP, seed=FIXED_TEST_SEED):
    """Fixed random sample of at most `cap` test rows (identical for every model)."""
    if len(test_idx) <= cap:
        return np.asarray(test_idx)
    return np.sort(np.random.RandomState(seed).choice(test_idx, cap, replace=False))


def time_split(df):
    """Train on the earliest 80% of calendar dates, test on the latest 20%.

    Rows after TIME_SPLIT_LAST_DATE are dropped first, so that the test period
    covers all nine routes. No date appears on both sides. The test side is then
    reduced to a fixed sample of at most FIXED_TEST_CAP rows. Deterministic.
    """
    date = df["date"].values
    keep = date <= np.datetime64(TIME_SPLIT_LAST_DATE)
    dates = np.sort(pd.unique(date[keep]))
    n_train_dates = int(np.floor(TIME_SPLIT_TRAIN_FRAC * len(dates)))
    boundary = dates[n_train_dates]              # first test date
    train_idx = np.where(keep & (date < boundary))[0]
    test_idx = np.where(keep & (date >= boundary))[0]
    return train_idx, cap_test(test_idx)


def get_split(df, split_type, seed):
    """Same splitting rules as run_experiment in notebooks/Experiments.ipynb, plus "time"."""
    n = len(df)
    if split_type == "trip_grouped":
        return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(np.zeros(n), groups=df["trip_id"].values))
    if split_type == "route_grouped":
        return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(np.zeros(n), groups=df["route_id"].values))
    if split_type == "random":
        return train_test_split(np.arange(n), test_size=0.2, random_state=seed, shuffle=True)
    if split_type == "time":
        return time_split(df)
    raise ValueError(f"Unknown split_type: {split_type!r}")


def sample_whole_trips(groups, target_rows, rng):
    """Add whole trips in random order until at least `target_rows` rows are selected.

    Gives the same rows as sample_by_group in notebooks/Ferry_Delay_Regression.ipynb
    (Step 9) for the same `rng`, without the per-trip loop.
    """
    trips, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(trips))
    n_trips = int(np.searchsorted(np.cumsum(counts[order]), target_rows)) + 1
    return np.where(np.isin(groups, trips[order[:n_trips]]))[0]


def sample_trips_exact(groups, k, rng):
    """Draw exactly `k` rows trip by trip: whole trips in random order, with the
    last trip cut after its first stops so the total is exactly `k`.

    Returns (row positions, number of distinct trips). Rows of a trip are in
    stop-sequence order in the cleaned table, so a cut trip keeps its earliest stops.
    """
    if k == 0:
        return np.array([], dtype=int), 0
    trips, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = rng.permutation(len(trips))
    cum = np.cumsum(counts[order])
    n_trips = int(np.searchsorted(cum, k)) + 1
    if n_trips > len(trips):
        raise ValueError(f"only {cum[-1]} rows available, {k} requested")
    whole = np.where(np.isin(inverse, order[:n_trips - 1]))[0]
    last = np.where(inverse == order[n_trips - 1])[0][:k - len(whole)]
    return np.sort(np.concatenate([whole, last])), n_trips
