"""Model comparison under three validation schemes of increasing difficulty.

random         : 10 random 70/30 splits (the setting used in most papers).
leave-additive : each of the 22 additives is held out entirely (unseen additive).
leave-core     : each of the 5 aryl-halide cores is held out entirely.

Out-of-fold predictions for the grouped schemes are pooled before scoring.
Only scikit-learn is required; the one-hot design cannot see unseen categories,
which is exactly the limitation the grouped schemes are meant to expose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

SEED = 0
REPORTS = data.ROOT / "reports"


def make_models():
    return {
        "mean baseline": DummyRegressor(strategy="mean"),
        "ridge (one-hot)": Ridge(alpha=1.0),
        "random forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "hist. gradient boosting": HistGradientBoostingRegressor(random_state=SEED),
    }


def scores(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run_schemes(df: pd.DataFrame, X: pd.DataFrame, models_fn=make_models):
    """Score every model from models_fn() under the three validation schemes.

    X is the design matrix for the rows of df. Returns (metrics, oof_predictions).
    """
    y = df["yield_pct"].to_numpy()
    rows, oof_frames = [], []

    # 1) random 70/30, 10 repeats -> mean +- std over repeats
    splitter = ShuffleSplit(n_splits=10, test_size=0.3, random_state=SEED)
    for name in models_fn():
        per_split = []
        for tr, te in splitter.split(X):
            m = models_fn()[name].fit(X.iloc[tr], y[tr])
            per_split.append(scores(y[te], m.predict(X.iloc[te])))
        res = pd.DataFrame(per_split)
        rows.append({"scheme": "random 70/30", "model": name,
                     **{k: res[k].mean() for k in res}, **{k + "_std": res[k].std() for k in res}})

    # 2) + 3) grouped: pooled out-of-fold predictions
    for scheme, group_col in [("leave-additive-out", "additive_id"), ("leave-core-out", "core_id")]:
        groups = df[group_col].to_numpy()
        logo = LeaveOneGroupOut()
        for name in models_fn():
            oof = np.empty(len(df))
            for tr, te in logo.split(X, y, groups):
                m = models_fn()[name].fit(X.iloc[tr], y[tr])
                oof[te] = m.predict(X.iloc[te])
            rows.append({"scheme": scheme, "model": name, **scores(y, oof)})
            oof_frames.append(pd.DataFrame({"scheme": scheme, "model": name,
                                            "row": np.arange(len(df)), "y_true": y, "y_pred": oof,
                                            "group": groups}))
    return pd.DataFrame(rows), pd.concat(oof_frames)


def main() -> None:
    df = data.load_processed()
    X = data.onehot(df)
    REPORTS.mkdir(exist_ok=True)

    metrics, oof = run_schemes(df, X)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)
    oof.to_csv(REPORTS / "oof_predictions.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(metrics[["scheme", "model", "R2", "RMSE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
