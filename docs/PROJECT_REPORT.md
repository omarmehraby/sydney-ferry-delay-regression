# Ferry Arrival Delay — Project Report
### TabPFN vs. XGBoost vs. LightGBM — Sydney Ferries Regression Study (ICCL26)

*Generated 2026-08-26. Covers all work completed to date across
`Ferry_Delay_Regression.ipynb` (the main pipeline) and `Experiments.ipynb`
(the experiment sandbox).*

> **Note (9 October 2026).** This report describes the project as of August
> 2026. The TabPFN numbers in it come from the model the hosted service served
> then. Later work, including re-runs on the current model, a time-based split
> and a few-shot route experiment, is in `docs/PAPER_NOTES.md` and `paper/`.

---

## 1. Project overview

**Goal:** predict Sydney ferry `arrival_delay` (seconds late/early) from trip and
weather features, comparing three regression models — XGBoost, LightGBM, and
TabPFN (hosted API, ~10,000-row context limit) — on accuracy and runtime, with a
specific focus on **proving results aren't caused by data leakage** (the
professor's key methodology requirement, carried over from the team's prior SURP
project where a repeating categorical entity was found to be leaking between
train and test).

**Two files, two purposes:**
- `Ferry_Delay_Regression.ipynb` — the finished, presentable pipeline. Never
  modified except by explicit request; every change to it was deliberate and is
  logged below.
- `Experiments.ipynb` — a sandbox added later for controlled experiments (feature
  ablations, temporal features, multi-seed robustness checks) that logs every
  result to `experiment_results.csv`, without ever touching the main pipeline.

---

## 2. Data and cleaning pipeline

- **Source:** `Data-Sydney 1.csv` — 1,048,575 rows × 27 columns.
- **⚠️ Likely truncated:** the row count is exactly Excel's 1,048,576-row limit
  (minus header), and two stray columns (`Unnamed: 26`, `1048576`) contain
  literal Excel row-index values. **This should be confirmed with Prof. Sarhani
  before finalizing any paper claims** — it has not been resolved yet.
- **De-duplication:** the source data was weather-joined in a way that doubled
  every row. Grouping on `(trip_id, start_date, current_stop_sequence, stop_id)`
  collapses this: **1,048,575 → 496,042 unique ferry events.**
- **Cleaning steps:** parse unit-suffixed weather strings (`"68 °F"`, `"62%"`) to
  numbers; convert Excel-serial/string dates and times to real datetimes;
  missing-value **indicator flags** added (`stop_id_missing`, `Wind_missing`,
  `Condition_missing`) instead of imputing; target **winsorized** to
  [-206, 508] seconds to remove impossible outliers (raw max was 44,575s ≈
  12 hours).
- **Target sanity check:** after cleaning, `arrival_delay` has **mean 66.1s,
  std 121.2s** — genuine spread, not a near-constant dead end (this was an
  explicit risk flagged from the team's prior SURP project, where a
  near-constant target was a dead end).
- **Features (21 total):** 6 categorical (`route_id`, `stop_id`, `vehicle_id`,
  `Wind`, `Condition`, `Class`) + 12 numeric (`start_time`,
  `start_date_ordinal`, `month`, `day_of_year`, `current_stop_sequence`, `Week`,
  `Dew Point`, `Humidity`, `Wind Speed`, `Pressure`, `Seating Capacity`,
  `Holidays`) + 3 missing-flags. `trip_id` is deliberately excluded as a feature
  (used only for splitting).
- **Models:** XGBoost and LightGBM at matched, **untuned** defaults
  (`n_estimators=300`, `learning_rate=0.1`, comparable depth/subsample settings)
  — fairness principle from the team's briefing. TabPFN via hosted API, wrapped
  to auto-subsample training data to its 10,000-row context limit whenever
  exceeded.

---

## 3. Baseline result (trip-grouped, all features)

The main pipeline's headline single-split result (80/20, grouped by `trip_id`
so no trip leaks between train/test), on the full 496,042-row dataset:

| Model | RMSE (s) | MAE (s) | R² | Fit time |
|---|---|---|---|---|
| XGBoost | 95.9 | 69.2 | 0.369 | 8.4s |
| LightGBM | 96.5 | 69.8 | 0.362 | 4.6s |
| TabPFN | 100.9 | 71.4 | 0.303 | 3.5s |

**Scaling experiment** (training size 1,000 → 100,000 rows, 3 seeds): RMSE
improves steadily for all three models as training size grows. TabPFN has the
lowest RMSE at every size from 1,000 to 50,000 rows except 2,000, where it is
about 1 s behind XGBoost and level with LightGBM (e.g. at 20,000 rows: RMSE
103.8 vs LightGBM 105.7 and XGBoost 106.5), even though it is capped at a
10,000-row subsample above 10,000 rows. It is overtaken only at 100,000 rows
(XGBoost/LightGBM RMSE ≈ 99.5 vs TabPFN ≈ 100.6). Boosters' fit
time stays under 4s throughout; TabPFN's fit time is dominated by API overhead
(2.4–6.8s) regardless of size — and this chart only measures fit time, not the
often-larger prediction-side API latency.

**Feature importance** (LightGBM permutation importance, chosen because it's
model-agnostic and cheap — computing it for TabPFN would mean thousands of extra
hosted-API calls): `stop_id` (0.200), `route_id` (0.161),
`current_stop_sequence` (0.138), and `start_time` (0.093) dominate; weather
features rank far lower.

---

## 4. Leakage investigation — the headline finding

The trip-grouped split above still lets the **same route and stop** appear in
both train and test — only the exact trip is separated. Since `stop_id` and
`route_id` are the two most important features, this was suspected of masking
memorization rather than transferable signal. Three splitting schemes were
compared, averaging the route-grouped split over 5 seeds since only **9 distinct
routes** exist in this dataset:

| Model | Random split (leaky) | Trip-grouped (honest default) | Route-grouped (stress test) |
|---|---:|---:|---:|
| XGBoost | R²=0.399 | R²=0.369 | **R²=−0.398** |
| LightGBM | R²=0.382 | R²=0.362 | **R²=−0.420** |
| TabPFN | R²=0.309 | R²=0.303 | **R²=−0.099** |

**Verdict: confirmed leakage.** Holding out entire routes drives R² negative for
every model — worse than always predicting the average delay. The trip-grouped
headline (R² ≈ 0.30–0.37) is optimistic; the models were substantially leaning
on memorized, per-route delay patterns. This is exactly the "big lesson learned"
failure pattern the professor's briefing warned about (there it was a "store"
entity in the SURP data; here it's `route_id`/`stop_id`), and is the single most
important methodological result in the project so far.

*(Full detail: `results/main_pipeline/REPORT.md`, `results/main_pipeline/split_comparison.csv/.png`.)*

---

## 5. Ablation study — quantifying the leakage driver

To test whether `route_id`/`stop_id` were actually causing the collapse, three
feature sets were compared under both trip-grouped and route-grouped splits:

- **A — all features** (21 features)
- **B — no `route_id`/`stop_id`** (18 features)
- **C — no identity features at all**, also drops `vehicle_id` (17 features)

| | Trip-grouped R² | | | Route-grouped R² | | |
|---|---:|---:|---:|---:|---:|---:|
| Model | A | B | C | A | B | C |
| XGBoost | 0.369 | 0.245 | 0.228 | −0.398 | −0.112 | −0.136 |
| LightGBM | 0.362 | 0.245 | 0.228 | −0.420 | −0.069 | −0.092 |
| TabPFN | 0.303 | 0.130 | 0.118 | −0.099 | −0.098 | −0.088 |

**Finding:** removing `route_id`/`stop_id` costs real in-distribution accuracy
(trip-grouped R² drops from ~0.34 avg to ~0.20 avg) but substantially **improves
generalization** to unseen routes — average route-grouped R² rises from −0.31 to
about −0.09 to −0.11. Still negative (only 9 routes exist, each behaving quite
differently, is a harder structural problem), but the gap shrinks a lot,
confirming `route_id`/`stop_id` were a real, quantifiable driver of the leakage
— not the whole story, but the dominant one. Dropping `vehicle_id` too (config C)
barely moves anything further beyond B.

*(Full detail: `results/ablation/ABLATION_REPORT.md`, `results/ablation/ablation_r2_comparison.png`.)*

---

## 6. Experiment tracking infrastructure

To keep exploring without ever touching the finished pipeline, `Experiments.ipynb`
was built as a self-contained sandbox: same data source and cleaning logic
(copied, not imported), a `FEATURE_SETS` registry, and one core function,
`run_experiment(experiment_name, feature_set, split_type, model_names, seed)`,
that trains, times (training and prediction separately), scores, and **appends**
one row per model to `experiment_results.csv` — verified in practice (ran twice,
row count doubled correctly, header not duplicated) to never overwrite prior
results. Columns: `experiment_name, feature_set, split_type, model, MAE, RMSE,
R2, training_time, prediction_time, random_seed`.

---

## 7. Temporal feature experiment

**Question:** can better time-of-day/seasonality features recover some of the
accuracy lost when identity features (B/C) are removed?

**New features**, added only to two new variants (`no_route_stop_temporal`,
`no_identity_temporal`), not to feature set A: `hour_of_day`, `day_of_week`,
`weekend_flag`, `morning_peak_flag`, `evening_peak_flag`, and cyclical
`month_sin`/`month_cos`.

| Feature set | Model | R² before | R² after (trip-grouped) | R² before | R² after (route-grouped) |
|---|---|---:|---:|---:|---:|
| B: no route/stop | XGBoost | 0.245 | 0.254 | −0.112 | −0.103 |
| B: no route/stop | LightGBM | 0.245 | 0.246 | −0.069 | −0.075 |
| B: no route/stop | TabPFN | 0.130 | 0.150 | −0.098 | −0.106 |
| C: no identity | XGBoost | 0.228 | 0.240 | −0.136 | −0.118 |
| C: no identity | LightGBM | 0.228 | 0.234 | −0.092 | −0.082 |
| C: no identity | TabPFN | 0.118 | 0.131 | −0.088 | −0.094 |

**Finding:** small, consistent improvement (R² up ~0.006–0.014 in most cases),
but it does not come close to closing the gap left by removing identity
features, and does essentially nothing to pull route-grouped R² positive. Better
temporal features sharpen the model's existing signal a little; the
leakage-driven collapse is a route-identity problem, largely independent of
time-of-day/seasonality signal.

*(Full detail: `results/experiments/temporal_comparison_report.md`, `temporal_comparison_summary.csv`.)*

---

## 8. Final robustness check — 5-seed trip-grouped evaluation

To confirm the single-seed numbers used everywhere above weren't a lucky/unlucky
draw, all five feature configurations were re-evaluated under 5 different
trip-grouped splits (seeds 42–46), all three models, no new features, no
tuning:

| Feature set | Model | Mean R² ± std | Mean RMSE ± std | Mean MAE ± std |
|---|---|---|---|---|
| 1. All features | XGBoost | **0.365 ± 0.006** | 96.59 ± 0.63 | 69.66 ± 0.34 |
| | LightGBM | 0.355 ± 0.007 | 97.33 ± 0.65 | 70.35 ± 0.33 |
| | TabPFN | 0.300 ± 0.006 | 101.45 ± 0.70 | 71.97 ± 0.50 |
| 2. No route/stop | XGBoost | 0.244 ± 0.003 | 105.43 ± 0.49 | 77.75 ± 0.35 |
| | LightGBM | 0.238 ± 0.006 | 105.81 ± 0.82 | 78.12 ± 0.55 |
| | TabPFN | 0.122 ± 0.007 | 113.57 ± 0.84 | 84.13 ± 0.49 |
| 3. No identity | XGBoost | 0.228 ± 0.004 | 106.49 ± 0.56 | 78.49 ± 0.35 |
| | LightGBM | 0.224 ± 0.004 | 106.80 ± 0.67 | 78.84 ± 0.45 |
| | TabPFN | 0.109 ± 0.007 | 114.41 ± 0.82 | 84.92 ± 0.50 |
| 4. No route/stop + temporal | XGBoost | 0.251 ± 0.003 | 104.94 ± 0.59 | 77.32 ± 0.40 |
| | LightGBM | 0.240 ± 0.004 | 105.65 ± 0.69 | 78.01 ± 0.48 |
| | TabPFN | 0.147 ± 0.006 | 111.97 ± 0.87 | 82.49 ± 0.64 |
| 5. No identity + temporal | XGBoost | 0.238 ± 0.003 | 105.82 ± 0.55 | 77.94 ± 0.35 |
| | LightGBM | 0.227 ± 0.005 | 106.55 ± 0.66 | 78.63 ± 0.40 |
| | TabPFN | 0.126 ± 0.006 | 113.36 ± 0.75 | 83.89 ± 0.48 |

**Findings:**
- **Standard deviations are tiny** (R² std ≤ 0.007 everywhere, including TabPFN)
  — every earlier single-seed result in this project was representative, not a
  fluke.
- **All-features config is clearly best** in-distribution (R² ~0.30–0.36 vs
  ~0.11–0.25 for every reduced config) — consistent with the ablation study.
- **Temporal features give a small, consistent, statistically stable lift**
  (configs 2→4 and 3→5, across all three models) — confirms Section 7's finding
  with proper error bars instead of a single split.
- **XGBoost beats LightGBM in every single row**, by a small but consistent
  margin (~0.01 R²) — a stable ranking.
- **TabPFN loses noticeably more R² than the boosters when identity features are
  removed** — dropping from 0.300 (all features) to roughly 0.11–0.15 (a fall of
  ~0.15–0.19) versus XGBoost's smaller ~0.11–0.14 drop, suggesting TabPFN relies
  even more heavily on `route_id`/`stop_id` than the gradient boosters do.

*(Full detail: `results/experiments/final_comparison_table.csv`, `final_comparison_r2.png`.)*

---

## 9. Known issues, caveats, and open items

- **Source data likely truncated** (Section 2) — needs confirmation from Prof.
  Sarhani; unresolved.
- **Only 9 distinct routes** in the data, heavily imbalanced (`F3` alone is
  ~41% of raw rows) — the route-grouped result, while averaged over 5 seeds, is
  still based on a small number of held-out groups per split. Treat the exact
  magnitude as indicative; the *sign* (negative, consistently) is the reliable
  part.
- **Feature importance computed on LightGBM only**, not per-model — a pragmatic
  cost tradeoff (permutation importance on TabPFN would mean thousands of extra
  API calls), documented but still a real gap if asked "does XGBoost/TabPFN
  weight features the same way?"
- **TabPFN's hosted-API daily quota was exhausted mid-project** (hit a 429 with
  message "resets at 2026-08-24 00:00:00 UTC"), so the TabPFN runs for the
  temporal-feature experiment (Section 7) and the final 5-seed check (Section 8)
  were completed after the boosters. Those rows are now in
  `experiment_results.csv` (24 and 25 TabPFN rows respectively) and are included
  in the tables above. The quota remains a constraint on any new TabPFN
  experiment.
- **Runtime chart in Section 3 only measures fit time**, not TabPFN's
  prediction-side API latency, which in practice is often larger — the plotted
  runtime curve understates TabPFN's true wall-clock cost.
- `experiment_results.csv` currently has 149 rows and no duplicates on
  `(experiment_name, feature_set, split_type, model, random_seed)`. Re-running
  an experiment cell appends its rows again, so de-duplicate on that key before
  any analysis.

---

## 10. Suggested next steps before the conference

1. **Resolve the data-truncation question** with Prof. Sarhani — this affects
   every number in the paper if the file turns out to be incomplete.
2. ~~Fill in the missing TabPFN rows for the temporal-feature and final 5-seed
   experiments.~~ Done — see Sections 7 and 8.
3. **Lead with the leakage finding** (Section 4) as the paper's central
   methodological contribution, not a footnote — it's the most interesting and
   defensible result here, directly answering the professor's core requirement.
4. Consider **running the scaling experiment (Section 3) under the
   route-grouped split**, not just trip-grouped, to see whether more data helps
   the route-holdout case or if it's structurally stuck negative regardless of
   scale.
5. **Always de-duplicate `experiment_results.csv`** on `(experiment_name,
   feature_set, split_type, model, random_seed)` before analysis, since re-run
   cells append repeat rows (the file has none at present).

---

## Appendix — file index

| File | Contents |
|---|---|
| `notebooks/Ferry_Delay_Regression.ipynb` | Main pipeline: cleaning, baseline, scaling experiment, importance, leakage splits (Steps 11-13), ablation study (Step 15) |
| `notebooks/Experiments.ipynb` | Sandbox: experiment tracker, temporal-feature experiment, final 5-seed evaluation |
| `results/experiments/experiment_results.csv` | Append-only log of every tracked experiment |
| `results/main_pipeline/REPORT.md`, `results/main_pipeline/*.csv`, `results/main_pipeline/*.png` | Main pipeline's leakage report, plots, scaling/importance data |
| `results/ablation/ABLATION_REPORT.md`, `results/ablation/*.csv/.png` | Ablation study detail |
| `results/experiments/temporal_comparison_report.md`, `temporal_comparison_summary.csv` | Temporal-feature before/after detail |
| `results/experiments/final_comparison_table.csv`, `final_comparison_r2.png` | 5-seed final robustness check |
| `figures/` | Curated, numbered copies of the five headline figures |
| `docs/PROJECT_REPORT.md` | This document |
