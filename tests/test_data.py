import numpy as np
import pandas as pd
import pytest

import data


def test_prepare_adds_ids_halogen_and_core(reactions):
    out = data.prepare(reactions)
    assert set(out["halogen"]) == {"Cl", "Br", "I"}
    assert out["core_id"].nunique() == 5
    assert set(out["ligand_id"]) == {"L1", "L2"}
    assert not out["aryl_core"].str.startswith(("Cl", "Br")).any()  # halogen is stripped from the core


def test_prepare_ids_do_not_depend_on_row_order(reactions):
    a = data.prepare(reactions)
    b = data.prepare(reactions.sample(frac=1.0, random_state=1).reset_index(drop=True))
    key = ["ligand", "additive", "base", "aryl_halide"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(a)
    for col in ["ligand_id", "base_id", "additive_id", "core_id"]:
        assert (merged[col + "_a"] == merged[col + "_b"]).all()


def test_prepare_rejects_duplicate_combinations(reactions):
    with pytest.raises(AssertionError):
        data.prepare(pd.concat([reactions, reactions.iloc[[0]]], ignore_index=True))


def test_prepare_rejects_missing_values(reactions):
    bad = reactions.copy()
    bad.loc[0, "yield_pct"] = np.nan
    with pytest.raises(AssertionError):
        data.prepare(bad)


def test_prepare_rejects_yield_out_of_range(reactions):
    bad = reactions.copy()
    bad.loc[0, "yield_pct"] = 120.0
    with pytest.raises(AssertionError):
        data.prepare(bad)


def test_prepare_rejects_aryl_without_halogen(reactions):
    bad = reactions.copy()
    bad.loc[0, "aryl_halide"] = "FC1=CC=CC=C1"
    with pytest.raises(AssertionError):
        data.prepare(bad)


def test_onehot_has_one_active_column_per_component(reactions):
    X = data.onehot(data.prepare(reactions))
    assert (X.sum(axis=1) == 5).all()  # ligand, base, additive, core, halogen
    assert X.isin([0.0, 1.0]).all().all()
