# Notes for the paper

Points that must be stated in the paper, and decisions already taken. Each item
names the section it belongs in. Numbers are not repeated here; take them from
the results files.

## Framing

*(Abstract, Introduction, Conclusion.)* TabPFN's data efficiency is what makes
it suited to new routes. Three contributions:

1. **Evaluation ladder.** Random, trip-grouped, time, unseen routes: the
   boosted models score lower on each, and every model is below zero on unseen
   routes. *Wording:* the ladder holds for XGBoost and LightGBM. TabPFN does not
   follow it, since its time-split R2 is higher than its trip-grouped R2, so say
   "the boosted models" when describing the ladder. The time split is also
   scored on a different test set from the other splits.
2. **Few-shot routes.** TabPFN beats the route's own mean delay from about 100
   rows; the boosters need about 500 and stay behind up to 5,000. Name the
   exceptions (F4 at k = 100 and 500; F1, F2 and F7 at k = 0).
3. **Data efficiency.** See "Central claim" below.

State plainly: all results use default settings, and with zero rows from a
route no model beats the global mean. *Wording:* the second statement is exact
on the unseen-routes split (Table 2 of the paper), where the training global
mean has the highest mean R2. In the few-shot experiment at k = 0 every model
is below zero, but TabPFN and the stop-mean baseline are marginally above the
other routes' global mean, so there the paper says "no model is useful" and
gives the three numbers.

## Submission format

Review is not anonymous. The paper carries the three authors (Al Akhawayn
University, Ifrane; department still to be filled in), the repository link, an
acknowledgements placeholder and the competing-interests statement, and it
describes the two Sarhani et al. papers as the authors' own earlier work. The
limit is 8 pages including references.

## Venue history

The ICCL 2026 presentation of this project was abstract-only, with no published
paper. There is therefore nothing to cite for it, and this paper is the first
written account of the work.

## Software and run dates

*(Reproducibility paragraph.)*

- TabPFN was called through the hosted API with `tabpfn-client` 0.4.1 and
  `model_path="auto"` (the client default), which lets the server choose the
  model. On 9 October 2026 the settings endpoint reported `v3.5` as the default
  model version and `train_set_max_rows` = 1,000,000 for it (v2 and v2.5:
  50,000; v2.6: 100,000; v3: 1,000,000).
- **The 10,000-row cap is this project's restriction, not the service's.** Write
  "we restricted TabPFN to a random 10,000-row sample", never "TabPFN's
  10,000-row limit". TabPFN was not run on more rows, so the paper cannot say
  how it would do with all the data.
- **The August and October TabPFN results may come from different model
  versions.** The version served in August was not recorded. In the paper's
  split table the random, trip-grouped and unseen-route TabPFN columns are from
  August; the time column, both few-shot tables and the few-shot figure are from
  9 October. Either re-run the August TabPFN results with the current model or
  state this prominently.
- Hollmann et al. 2025 describes an earlier TabPFN version than v3.5; the paper
  needs the right reference for the version served (TODO).
- The equal-data result is to be presented as consistent with Hollmann et al.
  2025, not as new.
- TabPFN runs for the time split, the few-shot experiment and the
  time-respecting few-shot experiment: 9 October 2026.
- TabPFN runs for the random, trip-grouped and unseen-route splits (main
  notebook and the 5-seed evaluation): August 2026, exact dates not recorded.
- Package versions for everything else are in `requirements.txt`.

## Update after the 9 October re-runs (supersedes the August TabPFN figures)

All TabPFN results in the paper are now from 9 October 2026 on model v3.5
(`results/experiments/tabpfn_run_log.csv` records the served version per run).
The August TabPFN rows remain in `experiment_results.csv` under their original
experiment names but are no longer reported; the re-runs are experiment
`tabpfn_rerun`. What changed:

- **TabPFN on 10,000 rows is level across the first three splits** (random,
  trip-grouped, time all about 0.33). The ladder holds for the boosted models
  only.
- **Unseen routes:** TabPFN now averages about zero, with a range that straddles
  zero across the five splits. So "every model is below zero" and "no model
  beats the global mean" are no longer exact. The paper says "no model is
  reliably better than predicting a mean delay".
- **Trip-grouped, full data:** the boosted models still lead TabPFN-on-10,000,
  by a smaller margin than in August.
- **Time split, more rows for TabPFN:** with 50,000 rows and with all rows
  TabPFN is ahead of the boosted models on the same rows (one run each). With
  10,000 rows it is at least as good as the boosted models on all rows.
- The cross-date limitation is removed from the paper.
- **Same rows, every size (added later on 9 October).** The paper's
  training-rows table compares the three models on identical training rows and
  the same 20,000 test events: time split at 10,000 (5 samples), 50,000
  (3 samples) and all rows (single run, not repeated by decision); trip-grouped
  split at 10,000 and all rows (seeds 42 and 43). TabPFN is ahead in every row.
  On the trip-grouped split with all rows its lead is much larger than on the
  time split, so the boosted models' lead in the split table exists only
  because TabPFN is restricted to 10,000 rows there, and the evaluation ladder
  applies to TabPFN too once it sees all rows.
- **Date-grouped split (incomplete).** Experiment `date_grouped`, split_type
  `date_grouped_fixed20k`: 20% of calendar dates held out at random, nothing
  dropped, scored on a fixed 20,000-row test sample. Purpose: to test whether
  TabPFN's all-rows result on the trip-grouped split (0.487) comes from
  same-day rows being in both training and test. Done for seeds 42 and 43:
  baselines, boosters on all rows and on TabPFN's 10,000 rows. Done for seed 42
  only: TabPFN on 10,000 rows. **Not done: TabPFN on all rows**, because the
  API's daily limit (5,000,000 tokens, resets 00:00 UTC) was reached on
  9 October. Resume with `python src/run_tabpfn_reruns.py`; it skips what is
  logged. Until then: 0.487 stays out of the abstract and the contribution
  list, the data-efficiency claim leads with the time-split rows, and wherever
  0.487 appears the paper says the trip-grouped split shares calendar dates
  between training and test. After the run: if the all-rows gain shrinks, state
  date sharing as the likely reason and add the date-grouped column to the
  split table between trip-grouped and time (`DATE_COLUMN` in
  `src/make_paper_tables.py`); if not, report that and offer no cause.
- **Wording rules for these claims:** always "at default settings" and "on this
  dataset"; say that tuned boosted models and CatBoost were not tested; say
  that the few-shot experiments keep the 10,000-row restriction; report the
  prediction-time cost of all-rows TabPFN (about six minutes for 20,000 events
  through the hosted service, against a fraction of a second for the boosters).

Where the sections below quote August numbers (for example "0.36-0.37 vs
0.30"), read them with this update; the results files are authoritative.

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
   subsamples to 10,000 rows whenever the training set is larger (our
   restriction; see "Software and run dates"). Above 10,000
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
- **A `trip_id` is a scheduled service, not one sailing.** 43% of the 75,781
  trip identifiers occur on more than one date (up to six), giving 161,610
  sailings. So the "trip-grouped" split keeps a service's dates together, and
  the "distinct trips" logged in the few-shot experiment are distinct trip
  identifiers. In the time-respecting few-shot experiment a trip means one
  trip identifier on one date. Write "trip identifier" or "scheduled service"
  in the paper, not "trip", where the difference matters.
- **`Week` is the day of the week** (0 = Monday to 6 = Sunday), not a week
  number. The `no_date_features` follow-up therefore removed day of week along
  with the three calendar-position features. Describe it that way.
- **`Holidays` needs confirming.** It is 1 on 290 of 417 dates (about 70%), so
  it may mark ordinary days rather than holidays. Do not call it a holiday flag
  in the paper until the data provider confirms.
- `Precip.` is constant, `Hour` duplicates `start_time`, and `Ridership` is a
  monthly per-route total that would not be known at prediction time. None is
  used as a feature.
