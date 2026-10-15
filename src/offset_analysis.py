"""Is the poor score on unseen cores / additives mostly a level shift?

Hypothesis: for a group the model has never seen (a core or an additive), it does not
know that group's average yield, and R^2 computed on that group punishes exactly this
shift. Test: split each group's error into
    MSE = bias^2 + variance of the error,
where bias is the mean error (prediction - truth) on the held-out group, and recompute
R^2 after subtracting that bias from the predictions ("oracle offset removal").

The correction uses the held-out group's true mean, so it is a diagnostic, not a usable
model. If most of the error is bias and R^2 jumps after removing it, the model ranks
reactions within the group reasonably but misjudges its overall level.

Works on reports/oof_predictions.csv written by evaluate.py (one-hot feature set).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402


def per_group(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per (scheme, model, held-out group)."""
    rows = []
    for (scheme, model), sub in oof.groupby(["scheme", "model"]):
        # every row appears once as a held-out row, so these are totals over all reactions
        total_sum, total_n = sub["y_true"].sum(), len(sub)
        for group, g in sub.groupby("group"):
            err = g["y_pred"] - g["y_true"]
            bias = err.mean()
            mse = (err ** 2).mean()
            sst = ((g["y_true"] - g["y_true"].mean()) ** 2).sum()
            # leave-one-group-out: the training set is every other group
            train_mean = (total_sum - g["y_true"].sum()) / (total_n - len(g))
            rows.append({
                "scheme": scheme, "model": model, "group": group, "n": len(g),
                "mean_yield": g["y_true"].mean(), "bias": bias, "mse": mse,
                "bias_share_of_mse": bias ** 2 / mse,
                "r2_raw": 1 - (err ** 2).sum() / sst,
                "r2_offset_removed": 1 - ((err - bias) ** 2).sum() / sst,
                "spearman_within_group": g["y_true"].corr(g["y_pred"], method="spearman"),
                # level the model assigns to the held-out group vs. the mean it was trained on
                "mean_pred": g["y_pred"].mean(), "train_mean_yield": train_mean,
                # >1: predictions are more spread out than the true yields within this group
                "pred_to_true_std": g["y_pred"].std() / g["y_true"].std(),
            })
    return pd.DataFrame(rows)


def pooled(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per (scheme, model): pooled R^2 as in metrics.csv, and with per-group offsets removed.

    `r2_group_mean_only` is the R^2 of a predictor that knows only each group's true mean
    (the share of yield variance that the group identity alone explains).
    """
    rows = []
    for (scheme, model), g in oof.groupby(["scheme", "model"]):
        bias = (g["y_pred"] - g["y_true"]).groupby(g["group"]).transform("mean")
        group_mean = g.groupby("group")["y_true"].transform("mean")
        rows.append({
            "scheme": scheme, "model": model,
            "r2_pooled": r2_score(g["y_true"], g["y_pred"]),
            "r2_pooled_offset_removed": r2_score(g["y_true"], g["y_pred"] - bias),
            "r2_group_mean_only": r2_score(g["y_true"], group_mean),
        })
    return pd.DataFrame(rows)


def main(oof_path=None, out_dir=None):
    oof_path = Path(oof_path) if oof_path else evaluate.REPORTS / "oof_predictions.csv"
    out_dir = Path(out_dir) if out_dir else evaluate.REPORTS
    if not oof_path.exists():
        raise SystemExit(f"{oof_path} not found: run src/evaluate.py first")
    oof = pd.read_csv(oof_path)
    oof = oof[oof["model"] != "mean baseline"]

    by_group, total = per_group(oof), pooled(oof)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(out_dir / "offset_analysis_by_group.csv", index=False)
    total.to_csv(out_dir / "offset_analysis_pooled.csv", index=False)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("Pooled R2: raw vs with per-group offset removed")
        print(total.to_string(index=False))
        print("\nLeave-core-out, by held-out core")
        print(by_group[by_group.scheme == "leave-core-out"].drop(columns="scheme").to_string(index=False))
    return by_group, total


if __name__ == "__main__":
    main()
