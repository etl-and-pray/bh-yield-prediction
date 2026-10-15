"""Loading and light feature engineering for the Buchwald-Hartwig HTE dataset
(Ahneman et al., Science 2018; table as distributed by Schwaller et al. in
rxn4chemistry/rxn_yields).

Every sheet in the source workbook holds the same 3955 reactions in a different
order, so only one sheet is read. Components are given as SMILES strings.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"
PROCESSED = ROOT / "data" / "processed" / "reactions.csv"

COMPONENTS = ["ligand", "additive", "base", "aryl_halide"]
_HALOGEN = re.compile(r"^(Cl|Br|I)(.+)$")


def _short_ids(series: pd.Series, prefix: str) -> pd.Series:
    """Stable readable ids (L1, L2, ...) assigned in alphabetical SMILES order."""
    mapping = {smi: f"{prefix}{i + 1}" for i, smi in enumerate(sorted(series.unique()))}
    return series.map(mapping)


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="FullCV_01")
    df = df.rename(
        columns={
            "Ligand": "ligand",
            "Additive": "additive",
            "Base": "base",
            "Aryl halide": "aryl_halide",
            "Output": "yield_pct",
        }
    )
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the table and add ids plus halide / aryl-core columns."""
    assert df.notna().all().all(), "unexpected missing values"
    assert not df.duplicated(COMPONENTS).any(), "duplicate reaction combinations"
    assert df["yield_pct"].between(0, 100).all(), "yield outside 0-100"

    out = df.copy()
    parsed = out["aryl_halide"].str.extract(_HALOGEN)
    assert parsed[0].notna().all(), "aryl halide SMILES does not start with Cl/Br/I"
    out["halogen"] = parsed[0]
    out["aryl_core"] = parsed[1]
    assert out["halogen"].nunique() == 3
    # 15 aryl halides must be 5 cores x 3 halogens
    combos = out[["aryl_core", "halogen"]].drop_duplicates()
    assert len(combos) == 15 and combos["aryl_core"].nunique() == 5

    out["ligand_id"] = _short_ids(out["ligand"], "L")
    out["base_id"] = _short_ids(out["base"], "B")
    out["additive_id"] = _short_ids(out["additive"], "A")
    out["core_id"] = _short_ids(out["aryl_core"], "C")
    return out


def load_processed() -> pd.DataFrame:
    """Prepared table; written to data/processed on first use."""
    df = prepare(load_raw())
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    return df


def onehot(df: pd.DataFrame, cols=("ligand_id", "base_id", "additive_id", "core_id", "halogen")) -> pd.DataFrame:
    """One-hot design matrix. Unseen categories at predict time become all-zero."""
    return pd.get_dummies(df[list(cols)], dtype=float)
