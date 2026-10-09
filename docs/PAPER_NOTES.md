# Notes for the paper

Points that must be stated in the paper, and decisions already taken. Each item
names the section it belongs in. Numbers are not repeated here; take them from
the results files.

## Central claim

**On equal data TabPFN beats both boosters; the boosters' full-data lead comes
from about 40 times more training rows.** *(Abstract, Results, Conclusion.)*
Evidence: experiments `equal_data_10k` and `final_5seed_trip_grouped` in
`results/experiments/experiment_results.csv` (trip-grouped split, feature set
A, seeds 42-46). TabPFN and the equal-data boosters are trained on the same
10,000 rows; the full-data boosters on about 396,000.

- **Limitation of the claim.** *(Limitations.)* It holds for default settings.
  Boosters tuned for small training sets were not tested, so the paper must not
  say that TabPFN beats gradient boosting on small data in general.
- **The existing scaling curve mixes row count and trip coverage.** *(Results;
  caption of the scaling figure.)* In `results/main_pipeline/scaling_results.csv`
  rows are sampled as whole trips, so a larger sample also covers more trips,
  and TabPFN's 10,000 rows cover more trips as the pool grows. The experiments
  `diversity_whole_trips_10k` and `diversity_random_from_100k_10k` separate the
  two effects for the boosters: with the row count fixed at 10,000, covering
  more trips improves accuracy. Either redraw the curve or state this.

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

## Facts to carry into Data and Limitations

- The source file has 1,048,575 rows, exactly Excel's limit, and its last month
  block (August 2023) contains one route only. It was probably truncated on
  export. *(Limitations.)*
- No rows exist for May, June or July 2022.
- Weather-join copies: 466,466 of 496,042 events have more than one copy (1 to
  14 copies), differing almost only in weather readings. De-duplication keeps
  the mean of the four numeric weather columns and the first copy's value for
  everything else. `arrival_delay` conflicts between copies in 30 events.
- `Precip.` is constant, `Hour` duplicates `start_time`, and `Ridership` is a
  monthly per-route total that would not be known at prediction time. None is
  used as a feature.
