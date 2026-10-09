# Paper

Springer LNCS format. `llncs.cls` and `splncs04.bst` are Springer's own files
(version 2.26, from CTAN) and must not be edited.

## Build

```bash
python src/make_paper_tables.py     # from the repository root; refreshes generated/
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload this folder together with `figures/6_fewshot_r2_vs_k.pdf`
and change the `\includegraphics` path in `main.tex` to match.

## Rules

- **No hand-typed result numbers.** Numbers in the text are macros defined in
  `generated/numbers.tex`; tables are in `generated/`. Both are written by
  `src/make_paper_tables.py` from the CSV files under `results/`. The dataset
  facts come from `results/dataset/`, written by `src/make_dataset_summary.py`
  (which needs the raw data file).
- **No unverified references.** Each entry in `references.bib` carries a comment
  saying where it was checked. Add an entry only after checking it against the
  publisher's or venue's own record.
- **`\todo{...}`** marks everything still to be written or confirmed. It prints
  in red so nothing is missed before submission.
- The points the paper must make, and the wording decisions, are in
  `docs/PAPER_NOTES.md`.

## Files

| File | Contents |
|---|---|
| `main.tex` | The paper |
| `references.bib` | Verified references only |
| `generated/numbers.tex` | Macros for numbers quoted in the text |
| `generated/tab_routes.tex` | Events, trips and stops per route |
| `generated/tab_splits.tex`, `tab_splits_rmse.tex` | R² and RMSE by model and split |
| `generated/tab_fewshot.tex` | Few-shot experiment, mean R² by k |
| `generated/tab_fewshot_routes.tex` | Few-shot experiment, R² per route |
