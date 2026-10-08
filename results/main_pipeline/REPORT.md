# Ferry Arrival Delay -- Leakage & Results Report

## What was added to the notebook
- **Step 11 -- naive (random) split**: a deliberately "wrong" 80/20 row-level shuffle,
  ignoring trip identity, to show the leaky/optimistic upper bound.
- **Step 12 -- route-grouped split**: held out entire routes (only 9
  distinct routes exist, so this is averaged over 5 different random splits
  for stability).
- **Step 13 -- summary table + bar chart** comparing all three splitting schemes
  side by side (`results/main_pipeline/split_comparison.csv`, `results/main_pipeline/split_comparison.png`).
- **Two writeup notes**: why feature importance used only LightGBM, and confirmation
  that `arrival_delay` has real spread (not a near-constant dead end).
- Every new result is both printed in its cell and saved under `results/main_pipeline/`:
  `naive_split_results.csv`, `route_split_by_seed.csv`, `split_comparison.csv`,
  `split_comparison.png`, `target_spread.csv`.
- Data source, cleaning, winsorising, missing-value indicators, and the existing
  XGBoost/LightGBM default settings were **not changed**.
- Used the **full cleaned dataset** (496,042) for all three splits --
  no sampling was needed, all comparisons ran in well under two minutes total.

## Results: R2 / RMSE under all three splits

| model    |   random_RMSE |   random_R2 |   trip_grouped_RMSE |   trip_grouped_R2 |   route_grouped_RMSE |   route_grouped_R2 |
|:---------|--------------:|------------:|--------------------:|------------------:|---------------------:|-------------------:|
| XGBoost  |        94.133 |       0.399 |              95.931 |             0.369 |              139.541 |             -0.398 |
| LightGBM |        95.445 |       0.382 |              96.503 |             0.362 |              140.547 |             -0.42  |
| TabPFN   |       100.945 |       0.309 |             100.893 |             0.303 |              124.749 |             -0.099 |

(RMSE and MAE are in seconds; lower is better. R2: higher is better, 0 = no
better than always guessing the average. A NEGATIVE R2 means worse than always
guessing the average.)

## Does the score depend on how we split the data? (the leakage verdict)
- **XGBoost**: random R2=0.399, trip-grouped R2=0.369, route-grouped R2=-0.398. Route-grouped drops notably (0.369 -> -0.398, -0.767) -- some reliance on route identity.
- **LightGBM**: random R2=0.382, trip-grouped R2=0.362, route-grouped R2=-0.420. Route-grouped drops notably (0.362 -> -0.420, -0.782) -- some reliance on route identity.
- **TabPFN**: random R2=0.309, trip-grouped R2=0.303, route-grouped R2=-0.099. Route-grouped drops notably (0.303 -> -0.099, -0.401) -- some reliance on route identity.

**Bottom line:** route-grouped R2 goes negative for all three models. This is a
real, large leakage signal -- the trip-grouped headline result overstates how well
these models would generalise to a route they have never seen.

## Which clues mattered most (from Step 10, LightGBM permutation importance)
1. `stop_id` (importance 0.200)
2. `route_id` (importance 0.161)
3. `current_stop_sequence` (importance 0.138)
4. `start_time` (importance 0.093)
5. `Seating Capacity` (importance 0.038)
6. `Class` (importance 0.031)
7. `Holidays` (importance 0.029)
8. `start_date_ordinal` (importance 0.020)

## Is `arrival_delay` a real signal or a near-constant dead end?
After cleaning and winsorising, `arrival_delay` has mean 66.1s and
standard deviation 121.2s across 496,042 rows -- genuine spread.
On the trip-grouped split R2 is roughly 0.30-0.37, so the *within-known-routes*
signal is real -- but see the leakage verdict above: that signal does not fully
carry over to brand-new routes.

## Warnings / things to double-check before showing your professor
- The source CSV's row count (1,048,575) is exactly Excel's row limit, and two
  stray columns (`Unnamed: 26`, `1048576`) contain literal Excel row-index values
  -- the file was very likely **truncated on export**. Confirm with Prof. Sarhani
  whether this is the complete dataset.
- Only 9 distinct routes exist, so the route-grouped
  result, while averaged over 5 splits, is still based on a small
  number of held-out groups per split -- treat the exact magnitude as indicative,
  not a high-precision estimate, but the sign (negative R2 across all 3 models,
  all 5 seeds' route combinations) is a consistent, not a one-off fluke.
- Feature importance was computed on LightGBM only, not per-model (see the writeup
  note above for why).
- **`stop_id` and `route_id` are the top two most important features, and the route-grouped R2 goes NEGATIVE for all three models** (average drop of 0.650 from the trip-grouped R2). A negative R2 means the models do *worse* than just guessing the average delay once an entire route is unseen. This is real evidence the models were leaning on memorised, route-specific delay patterns rather than a fully generalisable signal -- the exact failure mode the professor's leakage check was designed to catch. This is worth discussing directly with him: the trip-grouped headline numbers (R2 ~0.30-0.37) are likely optimistic, and the route-grouped numbers are the more honest, conservative estimate of how well this would generalise to a brand-new route.
