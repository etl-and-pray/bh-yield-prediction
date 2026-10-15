"""Grouped permutation importance: shuffle one whole reaction component in the
held-out set and measure the drop in R^2. Unlike per-column importance on a
one-hot matrix, this answers "how much does the model rely on the additive?"
as one number. Uses scikit-learn only (SHAP is a possible extension)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

SEED = 0
N_REPEATS = 20
GROUPS = {"ligand": "ligand_id", "base": "base_id", "additive": "additive_id",
          "aryl core": "core_id", "halogen": "halogen"}


def main() -> None:
    df = data.load_processed()
    X = data.onehot(df)
    y = df["yield_pct"].to_numpy()
    rng = np.random.default_rng(SEED)

    records = []
    for split_id, (tr, te) in enumerate(ShuffleSplit(n_splits=5, test_size=0.3, random_state=SEED).split(X)):
        model = HistGradientBoostingRegressor(random_state=SEED).fit(X.iloc[tr], y[tr])
        base_r2 = r2_score(y[te], model.predict(X.iloc[te]))
        for label, col in GROUPS.items():
            for _ in range(N_REPEATS):
                shuffled = df.copy()
                shuffled.loc[df.index[te], col] = rng.permutation(df[col].to_numpy()[te])
                Xp = data.onehot(shuffled).reindex(columns=X.columns, fill_value=0.0)
                records.append({"split": split_id, "component": label,
                                "r2_drop": base_r2 - r2_score(y[te], model.predict(Xp.iloc[te]))})

    res = (pd.DataFrame(records).groupby("component")["r2_drop"]
           .agg(["mean", "std"]).sort_values("mean", ascending=False))
    res.to_csv(data.ROOT / "reports" / "permutation_importance.csv")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
