# Ablation Study -- Effect of Removing Identity Features

## Setup
Three feature sets, each trained and evaluated identically to the rest of the
notebook (matched XGBoost/LightGBM defaults, TabPFN via hosted API, same
trip-grouped and route-grouped splitting methodology as Steps 7 and 12):

- **A -- all features**: the full 21-feature set.
- **B -- no route_id/stop_id**: 18 features.
- **C -- no identity features** (also drops vehicle_id): 17 features.

## Per-model results (R2)
- **XGBoost**: trip-grouped R2  A=0.369 -> B=0.245 -> C=0.228   |   route-grouped R2  A=-0.398 -> B=-0.112 -> C=-0.136
- **LightGBM**: trip-grouped R2  A=0.362 -> B=0.245 -> C=0.228   |   route-grouped R2  A=-0.420 -> B=-0.069 -> C=-0.092
- **TabPFN**: trip-grouped R2  A=0.303 -> B=0.130 -> C=0.118   |   route-grouped R2  A=-0.099 -> B=-0.098 -> C=-0.088

## Does removing identity features help generalisation to unseen routes?
Removing identity features **improves** generalisation to unseen routes: average route-grouped R2 rises from -0.305 (all features) to -0.105 (no identity) -- a gain of 0.200. This is consistent with `route_id`/`stop_id` having driven the negative route-grouped scores in Step 12: once the model can no longer look up a route's identity, it is forced to rely on transferable clues (time, weather, stop position), which generalise better even though the honest in-distribution (trip-grouped) score drops by 0.153.

## Files
- `ablation_results_long.csv` -- every (ablation x split x model) result, long format
- `ablation_table_trip_grouped.csv`, `ablation_table_route_grouped.csv` -- pivoted tables
- `ablation_r2_comparison.png` -- grouped bar chart, R2 by ablation, split side by side
