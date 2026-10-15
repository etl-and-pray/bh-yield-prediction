"""How stable is the comparison one-hot vs. RDKit descriptors?

evaluate_descriptors.py gives one number per cell of the table. This script repeats
the comparison on many different splits and reports paired differences, i.e. for
each split the score of a descriptor set minus the score of one-hot on the *same*
split, so that "easy" and "hard" splits cancel out.

additive : 30 random group splits, 6 of the 22 additives (25%) held out each time.
core     : leave-one-core-out, R^2 reported separately for each of the 5 cores.

Caveat: splits overlap, so they are not independent. The spread across splits shows
how much the result depends on which additives happen to be held out; it is not a
confidence interval for performance on new chemistry.

Needs RDKit (descriptor sets). Takes roughly 5-10 minutes. Writes reports/robustness_*.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import descriptors  # noqa: E402
import evaluate  # noqa: E402
import evaluate_descriptors  # noqa: E402

MODELS = ["ridge", "random forest", "hist. gradient boosting"]
N_SPLITS = 30
TEST_FRACTION = 0.25  # share of additives held out in each split
BASELINE = "one-hot"


def feature_sets(df: pd.DataFrame, featurizers: dict) -> dict:
    """name -> (design matrix, function returning fresh models for that matrix)."""

    def onehot_models():
        m = evaluate.make_models()
        return {"ridge": m["ridge (one-hot)"], "random forest": m["random forest"],
                "hist. gradient boosting": m["hist. gradient boosting"]}

    sets = {BASELINE: (data.onehot(df), onehot_models)}
    for name, featurizer in featurizers.items():
        X, desc_cols = descriptors.build_features(df, featurizer)
        sets[name] = (X, lambda cols=desc_cols: {k: v for k, v in evaluate_descriptors.scaled_models(cols).items()
                                                 if k in MODELS})
    return sets


def additive_splits(df, sets, n_splits=N_SPLITS, test_fraction=TEST_FRACTION) -> pd.DataFrame:
    y = df["yield_pct"].to_numpy()
    groups = df["additive_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_fraction, random_state=evaluate.SEED)
    rows = []
    for split_id, (tr, te) in enumerate(splitter.split(df, y, groups)):
        n_groups = len(set(groups[te]))
        for fs, (X, models_fn) in sets.items():
            for name, model in models_fn().items():
                model.fit(X.iloc[tr], y[tr])
                rows.append({"features": fs, "model": name, "split": split_id, "n_test_additives": n_groups,
                             "R2": r2_score(y[te], model.predict(X.iloc[te]))})
        print(f"additive splits: {split_id + 1}/{n_splits}", flush=True)
    return pd.DataFrame(rows)


def core_folds(df, sets) -> pd.DataFrame:
    y = df["yield_pct"].to_numpy()
    groups = df["core_id"].to_numpy()
    rows = []
    for tr, te in LeaveOneGroupOut().split(df, y, groups):
        core = groups[te][0]
        for fs, (X, models_fn) in sets.items():
            for name, model in models_fn().items():
                model.fit(X.iloc[tr], y[tr])
                rows.append({"features": fs, "model": name, "held_out_core": core,
                             "R2": r2_score(y[te], model.predict(X.iloc[te]))})
        print(f"core folds: held out {core}", flush=True)
    return pd.DataFrame(rows)


def paired_summary(res: pd.DataFrame, unit: str, baseline: str = BASELINE) -> pd.DataFrame:
    """Per model and feature set: R2(features) - R2(baseline) on the same split / core.

    win_rate is the share of splits (or cores) where the feature set beats the baseline.
    """
    out = []
    for model, g in res.groupby("model"):
        wide = g.pivot(index=unit, columns="features", values="R2")
        for fs in wide.columns:
            if fs == baseline:
                continue
            d = wide[fs] - wide[baseline]
            out.append({"model": model, "features": fs, "n_units": len(d), "mean_diff": d.mean(),
                        "sd_diff": d.std(), "win_rate": (d > 0).mean()})
    return pd.DataFrame(out)


def main(featurizers=None, n_splits=N_SPLITS, out_dir=None):
    if featurizers is None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            raise SystemExit("RDKit is required: pip install -r requirements-rdkit.txt")
        featurizers = {"rdkit physchem": descriptors.physchem, "morgan counts": descriptors.morgan_counts}
    out_dir = Path(out_dir) if out_dir else evaluate.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    df = data.load_processed()
    sets = feature_sets(df, featurizers)

    add = additive_splits(df, sets, n_splits=n_splits)
    core = core_folds(df, sets)
    add.to_csv(out_dir / "robustness_additive_splits.csv", index=False)
    core.to_csv(out_dir / "robustness_core_folds.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nAdditive splits: R2 across splits")
        print(add.groupby(["model", "features"])["R2"].agg(["mean", "std", "min", "max"]).to_string())
        print("\nAdditive splits: paired difference vs one-hot")
        print(paired_summary(add, "split").to_string(index=False))
        print("\nCore folds: paired difference vs one-hot")
        print(paired_summary(core, "held_out_core").to_string(index=False))
    return add, core


if __name__ == "__main__":
    main()
