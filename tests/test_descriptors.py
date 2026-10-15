"""Feature assembly, tested with a stand-in featurizer (no RDKit needed)."""
import pandas as pd
import pytest

import data
import descriptors


def fake_featurizer(smiles_list):
    rows = {s: {"n_c": s.count("C"), "n_n": s.count("N"), "n_o": s.count("O"), "constant": 1.0}
            for s in smiles_list}
    return pd.DataFrame.from_dict(rows, orient="index")


def test_drop_constant_removes_only_constant_columns():
    table = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
    assert list(descriptors.drop_constant(table).columns) == ["a"]


def test_attach_looks_up_descriptors_by_smiles(reactions):
    df = data.prepare(reactions)
    out = descriptors.attach(df, "additive", fake_featurizer(sorted(df["additive"].unique())), "add_")
    assert list(out.index) == list(df.index)
    assert out.columns.str.startswith("add_").all()
    for smiles, group in df.groupby("additive"):
        assert (out.loc[group.index, "add_n_c"] == smiles.count("C")).all()


def test_attach_fails_when_a_molecule_is_missing(reactions):
    df = data.prepare(reactions)
    table = fake_featurizer(sorted(df["additive"].unique())[:-1])
    with pytest.raises(AssertionError):
        descriptors.attach(df, "additive", table, "add_")


def test_build_features_shape_and_columns(reactions):
    df = data.prepare(reactions)
    X, desc_cols = descriptors.build_features(df, fake_featurizer)
    assert len(X) == len(df) and not X.isna().any().any()
    assert all(c.startswith(("add_", "core_")) for c in desc_cols)
    assert not any(c.endswith("constant") for c in X.columns)  # constant descriptors are dropped
    onehot_cols = [c for c in X.columns if c not in desc_cols]
    assert all(c.startswith(("ligand_id_", "base_id_", "halogen_")) for c in onehot_cols)
