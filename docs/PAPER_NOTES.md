# Notes for the paper

Points that must be stated in the paper, and decisions already taken. Each item
names the section it belongs in. Numbers are not repeated here; take them from
the results files.

## Central claim

*(Abstract, Results, Conclusion. Numbers below are R2 and come from
`results/experiments/experiment_results.csv`; regenerate them from the file
rather than copying them from here.)*

1. **On equal data TabPFN beats both boosters, on both splits.** Given the same
   10,000 training rows, TabPFN is ahead of XGBoost and LightGBM on the
   trip-grouped split and on the time split, on every seed. Evidence:
   experiments `equal_data_10k`, `final_5seed_trip_grouped` and `time_split`.
2. **With all data the picture depends on the split.** The boosters lead on the
   trip-grouped split (0.36-0.37 vs 0.30 for TabPFN) but not on the time split
   (0.31-0.32 vs 0.333 +/- 0.005), where TabPFN is **at least as good** with 37
   times fewer training rows. Say "at least as good", not "better", for the
   full-data comparison.
3. **The cause is not established.** Removing the four date features
   (`start_date_ordinal`, `day_of_year`, `month`, `Week`) did not change the
   time-split result (feature set `no_date_features`), so the date features are
   not the explanation. Do not offer a cause in the paper.

Limitations of the claim *(Limitations)*:

- One time cutoff: a single train/test boundary was evaluated.
- The full-data boosters were run once on the time split (it is deterministic),
  so their time-split figures have no spread; TabPFN's spread is over five
  random 10,000-row samples.
- Default settings only. Boosters tuned for small training sets were not
  tested, so the paper must not say that TabPFN beats gradient boosting on
  small data in general.

Related point *(Results; caption of the scaling figure)*: **the existing
scaling curve mixes row count and trip coverage.** In
`results/main_pipeline/scaling_results.csv` rows are sampled as whole trips, so
a larger sample also covers more trips, and TabPFN's 10,000 rows cover more
trips as the pool grows. The experiments `diversity_whole_trips_10k` and
`diversity_random_from_100k_10k` separate the two effects for the boosters: with
the row count fixed at 10,000, covering more trips improves accuracy. Either
redraw the curve or state this.

## Must be stated

1. **TabPFN's training-size axis is not the boosters' axis.** *(Evaluation
   Protocol, and the caption of any scaling figure.)* The TabPFN wrapper
   subsamples to 10,000 rows whenever the training set is larger. Above 10,000
   rows, "training size" for TabPFN means the size of the pool its 10,000 rows
   were drawn from, not the number of rows it was given. XGBoost and LightGBM
   use every row. In the full-data runs TabPFN therefore sees 2–4% of the
   training rows.

2. **Weather features are observed values.** *(Data and Preprocessing;
   Limitations.)* Dew point, humidity, wind, wind speed, pressure and condition
   are the readings for the departure hour. A deployed model would have to use
   forecasts, so the reported accuracy is an upper bound on what weather
   contributes.

3. **The TabPFN cells for the temporal and 5-seed experiments are a
   reconstruction.** *(Reproducibility statement, if the paper has one;
   otherwise the repository README.)* The code that produced those logged
   TabPFN rows was not saved. The cells now in `notebooks/Experiments.ipynb`
   generate the same 49 run keys, but have not been re-executed against the
   API, so the logged TabPFN values are not confirmed reproducible.

## Decisions

- **Time-based split.** Rows after 2023-07-31 are dropped for this split only,
  because the source file contains August 2023 data for route F3 alone. Training
  uses the earliest 80% of the remaining calendar dates and testing the latest
  20%, with no date on both sides. State this in Evaluation Protocol. All other
  experiments keep every row.
- **Fixed test sample.** From Stage 3 onward every model is scored on the same
  fixed sample of at most 20,000 test rows, to stay inside the TabPFN quota.
  Earlier trip-grouped, route-grouped and random-split results use the full test
  set, so the two groups of results are not on identical test rows.
- **Wording.** Describe the route-grouped result as "generalization to unseen
  routes", not "leakage".
- **No tuning.** Every model keeps its default settings throughout.
- **Few-shot rows are whole trips.** In the few-shot route experiment the k
  rows from the new route are drawn as whole trips, not random rows, because a
  real new route is observed trip by trip. The number of distinct trips is
  logged for each k.
  Trips are added whole in random order and the last one is cut after its first
  stops, so each k is exactly k rows.
- **Few-shot test sets.** At most 2,000 test rows per route, the same rows for
  every model, k and seed. The band in the few-shot figure is +/- 1 standard
  error across the nine routes.

## Facts to carry into Data and Limitations

- The source file has 1,048,575 rows, exactly Excel's limit, and its last month
  block (August 2023) contains one route only. It was probably truncated on
  export. *(Limitations.)*
- No rows exist for May, June or July 2022.
- Weather-join copies: 466,466 of 496,042 events have more than one copy (1 to
  14 copies), differing almost only in weather readings. De-duplication keeps
  the mean of the four numeric weather columns and the first copy's value for
  everything else. `arrival_delay` conflicts between copies in 30 events.
- **`Week` is the day of the week** (0 = Monday to 6 = Sunday), not a week
  number. The `no_date_features` follow-up therefore removed day of week along
  with the three calendar-position features. Describe it that way.
- **`Holidays` needs confirming.** It is 1 on 290 of 417 dates (about 70%), so
  it may mark ordinary days rather than holidays. Do not call it a holiday flag
  in the paper until the data provider confirms.
- `Precip.` is constant, `Hour` duplicates `start_time`, and `Ridership` is a
  monthly per-route total that would not be known at prediction time. None is
  used as a feature.
