"""Summarise the few-shot route experiment: tables and the R2-vs-k figure.

Run from the repository root:  python src/make_fewshot_report.py

Reads results/experiments/fewshot_results.csv and writes

    results/experiments/fewshot_per_route.csv   one row per route, condition, k, model
                                                (mean over seeds)
    results/experiments/fewshot_summary.csv     one row per condition, k, model
                                                (mean and spread over routes)
    figures/6_fewshot_r2_vs_k.png / .pdf

Only (route, condition, k, model) cells with every seed present are used, and a
model is summarised over routes only where all nine routes are complete, so a
partly finished TabPFN run cannot bias a mean.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import REPO_ROOT
from run_stage4 import SEEDS, all_jobs

RESULTS = REPO_ROOT / "results" / "experiments"
FIGURES = REPO_ROOT / "figures"
N_ROUTES = 9

# colour follows the model; marker and dash repeat the identity for greyscale print
STYLE = {
    "TabPFN":        dict(color="#1baf7a", marker="o", ls="-"),
    "XGBoost":       dict(color="#2a78d6", marker="s", ls="-"),
    "LightGBM":      dict(color="#eb6834", marker="^", ls="-"),
    "Ridge":         dict(color="#52514e", marker="D", ls="--"),
    "RouteStopMean": dict(color="#8a8984", marker="v", ls=":"),
}
LABEL = {"RouteStopMean": "Route × stop mean"}
PANELS = [("pooled", "Pooled: other 8 routes + k rows"),
          ("target_only", "Target-only: the k rows alone")]


def load(seeds=None):
    df = pd.read_csv(RESULTS / "fewshot_results.csv")
    df = df.drop_duplicates(["route", "condition", "k", "model", "random_seed"])
    if seeds is not None:
        df = df[df.random_seed.isin(seeds)]
    return df


def per_route(df, n_seeds):
    g = df.groupby(["route", "condition", "k", "model"])
    out = g.agg(n_seeds=("random_seed", "nunique"), n_target_trips=("n_target_trips", "mean"),
                n_train_rows=("n_train_rows", "mean"), n_test_rows=("n_test_rows", "first"),
                R2=("R2", "mean"), R2_seed_std=("R2", "std"), RMSE=("RMSE", "mean"),
                MAE=("MAE", "mean"), training_time=("training_time", "mean"),
                prediction_time=("prediction_time", "mean")).reset_index()
    return out[out.n_seeds == n_seeds]


def summary(routes_table):
    g = routes_table.groupby(["condition", "k", "model"])
    out = g.agg(n_routes=("route", "nunique"), R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                R2_median=("R2", "median"), R2_min=("R2", "min"), R2_max=("R2", "max"),
                RMSE_mean=("RMSE", "mean"), MAE_mean=("MAE", "mean"),
                n_target_trips_mean=("n_target_trips", "mean")).reset_index()
    return out[out.n_routes == N_ROUTES]


def figure(summ, k_values):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b9b8b2", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    xpos = {k: i for i, k in enumerate(k_values)}
    for ax, (condition, title) in zip(axes, PANELS):
        sub = summ[summ.condition == condition]
        for model, st in STYLE.items():
            s = sub[sub.model == model].sort_values("k")
            if s.empty:
                continue
            x = s.k.map(xpos).values
            se = s.R2_std.values / np.sqrt(N_ROUTES)
            ax.fill_between(x, s.R2_mean - se, s.R2_mean + se, color=st["color"], alpha=0.13, lw=0)
            ax.plot(x, s.R2_mean, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                    ms=5.5, mec="white", mew=1.0, label=LABEL.get(model, model))
        ax.axhline(0, color="#0b0b0b", lw=0.8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(k_values)), [f"{k:,}" for k in k_values])
        ax.set_xlabel("k = rows observed on the new route")
        ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
    axes[0].set_ylabel("R² on the new route (mean of 9 routes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"6_fewshot_r2_vs_k.{ext}", dpi=300)
    plt.close(fig)


def main():
    df = load()
    # use only seeds that are complete for every model, so all models share the same draws
    have = set(df[["route", "condition", "k", "model", "random_seed"]].itertuples(index=False, name=None))
    expected = all_jobs(sorted(df.route.unique()), run_tabpfn=True)
    complete = [s for s in SEEDS if all(j in have for j in expected if j[4] == s)]
    if not complete:
        print("No seed is complete for TabPFN yet: summarising the other models only.")
        df = df[df.model != "TabPFN"]
        complete = [s for s in SEEDS
                    if all(j in have for j in expected if j[4] == s and j[3] != "TabPFN")]
    df = df[df.random_seed.isin(complete)]
    print(f"seeds used: {complete}")
    routes_table = per_route(df, n_seeds=len(complete))
    summ = summary(routes_table)
    routes_table.round(4).to_csv(RESULTS / "fewshot_per_route.csv", index=False)
    summ.round(4).to_csv(RESULTS / "fewshot_summary.csv", index=False)
    figure(summ, sorted(df.k.unique()))
    wide = summ.pivot_table(index=["condition", "model"], columns="k", values="R2_mean").round(3)
    print(wide.to_string())


if __name__ == "__main__":
    main()
