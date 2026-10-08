# Data

The raw dataset is **not stored in this repository** — it is 223 MB (over
GitHub's 100 MB per-file limit) and is not ours to redistribute.

## What to put here

Place the source file in this folder with exactly this name:

```
data/Data-Sydney 1.csv
```

Both notebooks read it from `../data/Data-Sydney 1.csv`. Ask a project member
for the file.

## What the file looks like

- 1,048,575 rows × 27 columns: Sydney Ferries stop-level arrival events joined
  with hourly weather.
- Key columns: `trip_id`, `route_id`, `stop_id`, `vehicle_id`, `start_time`,
  `start_date`, `current_stop_sequence`, `arrival_delay` (target, seconds),
  plus weather (`Dew Point`, `Humidity`, `Wind`, `Wind Speed`, `Pressure`,
  `Condition`) and vessel fields (`Class`, `Seating Capacity`).
- After de-duplication the pipeline works with 496,042 unique ferry events.

## Known issue

The row count equals Excel's row limit, and two stray columns hold Excel row
indices, so the file was probably truncated on export. This is unresolved —
see Section 9 of [docs/PROJECT_REPORT.md](../docs/PROJECT_REPORT.md).
