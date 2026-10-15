"""Figures for the README. Run after evaluate.py (needs reports/metrics.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

FIG = data.ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9.5, "figure.dpi": 140,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def yield_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(df["yield_pct"], bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlabel("Yield, %")
    ax.set_ylabel("Number of reactions")
    ax.set_title("Yields are skewed toward low values")
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.92, f"median {df['yield_pct'].median():.0f}%  |  zero yield {np.mean(df['yield_pct'] == 0):.0%}"
            f"  |  ≥70% {np.mean(df['yield_pct'] >= 70):.0%}", transform=ax.transAxes, ha="right", color=INK2)
    save(fig, "01_yield_distribution.png")


def component_effects(df):
    panels = [("halogen", "Halogen of aryl halide"), ("base_id", "Base"),
              ("ligand_id", "Ligand"), ("core_id", "Aryl core")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        m = df.groupby(col)["yield_pct"].mean().sort_values(ascending=False)
        ax.bar(m.index, m.values, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
        for x, v in enumerate(m.values):
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", color=INK2, fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean yield, %")
    fig.suptitle("Main effects of each reaction component (means over all other components)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_component_effects.png")


def additive_effects(df):
    m = df.groupby("additive_id")["yield_pct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(m.index, m.values, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    for y, v in enumerate(m.values):
        ax.text(v + 0.5, y, f"{v:.0f}", va="center", color=INK2, fontsize=8.5)
    ax.set_xlabel("Mean yield, %")
    ax.set_title("Additives differ by a factor of five in mean yield")
    ax.grid(axis="y", visible=False)
    save(fig, "03_additive_effects.png")


def ligand_base_heatmap(df):
    p = df.pivot_table(index="ligand_id", columns="base_id", values="yield_pct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(p.values, cmap="Blues", vmin=0, vmax=60)
    ax.set_xticks(range(p.shape[1]), p.columns)
    ax.set_yticks(range(p.shape[0]), p.index)
    ax.grid(False)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 35 else INK, fontweight="bold")
    ax.set_title("Mean yield by ligand × base")
    fig.colorbar(im, ax=ax, label="Mean yield, %", shrink=0.85)
    save(fig, "04_ligand_base_heatmap.png")


def validation_schemes(metrics):
    schemes = ["random 70/30", "leave-additive-out", "leave-core-out"]
    models = ["mean baseline", "ridge (one-hot)", "random forest", "hist. gradient boosting"]
    colors = [GRAY, BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    w = 0.2
    for k, (mod, col) in enumerate(zip(models, colors)):
        vals = [metrics[(metrics.scheme == s) & (metrics.model == mod)]["R2"].iloc[0] for s in schemes]
        xs = np.arange(len(schemes)) + (k - 1.5) * w
        ax.bar(xs, vals, width=w - 0.02, color=col, edgecolor=SURFACE, linewidth=1.2, label=mod)
        for x, v in zip(xs, vals):
            ax.text(x, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(schemes)), ["random 70/30\n(seen components)", "leave-additive-out\n(unseen additive)",
                                         "leave-core-out\n(unseen aryl core)"])
    ax.set_ylabel("R² on held-out reactions")
    ax.set_title("Accuracy collapses as the validation gets more realistic")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "05_validation_schemes.png")


def parity(oof):
    sub = oof[(oof.scheme == "leave-additive-out") & (oof.model == "hist. gradient boosting")]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(sub["y_true"], sub["y_pred"], s=7, color=BLUE, alpha=0.35, linewidths=0)
    ax.plot([0, 100], [0, 100], color=INK2, linewidth=1, linestyle="--")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-15, 103)
    ax.set_xlabel("Measured yield, %")
    ax.set_ylabel("Predicted yield, % (additive held out)")
    ax.set_title("Gradient boosting, unseen additives")
    save(fig, "06_parity_leave_additive_out.png")


def main():
    df = data.load_processed()
    yield_distribution(df)
    component_effects(df)
    additive_effects(df)
    ligand_base_heatmap(df)
    metrics = pd.read_csv(data.ROOT / "reports" / "metrics.csv")
    validation_schemes(metrics)
    parity(pd.read_csv(data.ROOT / "reports" / "oof_predictions.csv"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
