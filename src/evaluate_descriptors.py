"""Same three validation schemes as evaluate.py, but the additive and the aryl core
are described by molecular descriptors instead of one-hot columns.

Feature sets compared (results go to reports/metrics_descriptors.csv):
  rdkit physchem : size / polarity / atom-count descriptors
  morgan counts  : Morgan fingerprint substructure counts (radius 2, 128 bits)
The one-hot baseline is in reports/metrics.csv (run evaluate.py first).

Ridge standardizes the descriptor columns inside each training fold, so no
information from the held-out molecule leaks into the scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import descriptors  # noqa: E402
import evaluate  # noqa: E402


def scaled_models(desc_cols):
    base = evaluate.make_models()
    ridge = make_pipeline(
        ColumnTransformer([("scale", StandardScaler(), list(desc_cols))], remainder="passthrough"),
        Ridge(alpha=1.0),
    )
    return {
        "mean baseline": base["mean baseline"],
        "ridge": ridge,
        "random forest": base["random forest"],
        "hist. gradient boosting": base["hist. gradient boosting"],
    }


def main(featurizers=None, out_path=None) -> pd.DataFrame:
    if featurizers is None:
        featurizers = {"rdkit physchem": descriptors.physchem, "morgan counts": descriptors.morgan_counts}
    out_path = Path(out_path) if out_path else evaluate.REPORTS / "metrics_descriptors.csv"
    df = data.load_processed()

    frames = []
    for name, featurizer in featurizers.items():
        X, desc_cols = descriptors.build_features(df, featurizer)
        print(f"{name}: {X.shape[1]} features ({len(desc_cols)} descriptor columns)", flush=True)
        metrics, _ = evaluate.run_schemes(df, X, models_fn=lambda cols=desc_cols: scaled_models(cols))
        metrics.insert(0, "features", name)
        frames.append(metrics)

    result = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(result[["features", "scheme", "model", "R2", "RMSE", "MAE"]].to_string(index=False))
    return result


if __name__ == "__main__":
    main()
