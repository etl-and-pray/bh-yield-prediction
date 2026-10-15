import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

import data
import evaluate
import offset_analysis
import robustness


def test_paired_summary_math():
    res = pd.DataFrame({
        "features": ["one-hot"] * 3 + ["A"] * 3, "model": ["m"] * 6, "split": [0, 1, 2] * 2,
        "R2": [0.5, 0.6, 0.7, 0.6, 0.55, 0.9],
    })
    row = robustness.paired_summary(res, "split").iloc[0]
    assert np.isclose(row["mean_diff"], (0.1 - 0.05 + 0.2) / 3)
    assert np.isclose(row["win_rate"], 2 / 3)
    assert row["n_units"] == 3


def test_run_schemes_scores_every_reaction_once_per_grouped_scheme(reactions):
    df = data.prepare(reactions)
    X = data.onehot(df)
    metrics, oof = evaluate.run_schemes(df, X, models_fn=lambda: {"mean": DummyRegressor(), "ridge": Ridge()})
    assert set(metrics["scheme"]) == {"random 70/30", "leave-additive-out", "leave-core-out"}
    assert len(metrics) == 6 and metrics[["R2", "RMSE", "MAE"]].notna().all().all()
    for _, group in oof.groupby(["scheme", "model"]):
        assert sorted(group["row"]) == list(range(len(df)))  # each reaction held out exactly once


def test_offset_analysis_recovers_group_bias():
    rng = np.random.default_rng(0)
    parts = []
    for group, offset in {"C1": 10.0, "C2": -5.0, "C3": 0.0}.items():
        y = rng.uniform(0, 100, 60)
        parts.append(pd.DataFrame({"scheme": "leave-core-out", "model": "m", "group": group,
                                   "y_true": y, "y_pred": y + offset}))
    oof = pd.concat(parts, ignore_index=True)
    by_group = offset_analysis.per_group(oof).set_index("group")
    assert np.allclose(by_group["bias"], [10.0, -5.0, 0.0])
    assert np.allclose(by_group["r2_offset_removed"], 1.0)
    assert np.allclose(by_group.loc[["C1", "C2"], "bias_share_of_mse"], 1.0)
    pooled = offset_analysis.pooled(oof).iloc[0]
    assert np.isclose(pooled["r2_pooled_offset_removed"], 1.0) and pooled["r2_pooled"] < 1.0


def test_offset_analysis_mse_is_bias_squared_plus_error_variance():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 100, 200)
    pred = y + rng.normal(5, 10, 200)
    oof = pd.DataFrame({"scheme": "s", "model": "m", "group": ["g1"] * 100 + ["g2"] * 100,
                        "y_true": y, "y_pred": pred})
    row = offset_analysis.per_group(oof).iloc[0]
    err = (pred - y)[:100]
    assert np.isclose(row["mse"], err.mean() ** 2 + err.var(ddof=0))


def test_offset_analysis_training_mean_excludes_the_held_out_group():
    oof = pd.DataFrame({"scheme": "s", "model": "m", "group": ["a"] * 2 + ["b"] * 2,
                        "y_true": [10.0, 20.0, 30.0, 50.0], "y_pred": [0.0, 0.0, 0.0, 0.0]})
    by_group = offset_analysis.per_group(oof).set_index("group")
    assert np.isclose(by_group.loc["a", "train_mean_yield"], 40.0)  # mean of group b
    assert np.isclose(by_group.loc["b", "train_mean_yield"], 15.0)  # mean of group a
