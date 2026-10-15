"""Molecular descriptors for the additive and the aryl core (needs RDKit).

Idea: with one-hot encoding an unseen additive or core is an all-zero row, so the
model knows nothing about it. Describing each molecule by numbers (size, polarity,
atom counts, substructure counts) at least lets the model interpolate between
molecules it has seen. Ligand and base keep one-hot encoding (4 and 3 levels,
always present in training), and so does the halogen.

A featurizer is any function  list[str] -> DataFrame  indexed by SMILES. The two
RDKit ones are below; `build_features` accepts any other, which also makes the
plumbing testable without RDKit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def _mol(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def physchem(smiles_list) -> pd.DataFrame:
    """Ten physico-chemical descriptors plus heavy-atom counts per element."""
    from rdkit.Chem import Crippen, Descriptors
    from rdkit.Chem import rdMolDescriptors as rd

    rows = {}
    for smi in smiles_list:
        mol = _mol(smi)
        row = {
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rd.CalcTPSA(mol),
            "hbd": rd.CalcNumHBD(mol),
            "hba": rd.CalcNumHBA(mol),
            "rot_bonds": rd.CalcNumRotatableBonds(mol),
            "rings": rd.CalcNumRings(mol),
            "arom_rings": rd.CalcNumAromaticRings(mol),
            "frac_csp3": rd.CalcFractionCSP3(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
        }
        counts = {e: 0 for e in ELEMENTS}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in counts:
                counts[atom.GetSymbol()] += 1
        row.update({f"n_{e}": c for e, c in counts.items()})
        rows[smi] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def morgan_counts(smiles_list, radius: int = 2, n_bits: int = 128) -> pd.DataFrame:
    """Morgan (circular) fingerprint with substructure counts, folded to n_bits."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = {smi: gen.GetCountFingerprintAsNumPy(_mol(smi)).astype(float) for smi in smiles_list}
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"bit{i}" for i in range(n_bits)])


def drop_constant(table: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that do not vary across molecules (they carry no information)."""
    return table.loc[:, table.nunique() > 1]


def attach(df: pd.DataFrame, column: str, table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Look up each reaction's molecule (by SMILES in `column`) in the descriptor table."""
    feats = table.reindex(df[column].to_numpy())
    assert not feats.isna().any().any(), f"missing descriptors for some molecules in '{column}'"
    feats.index = df.index
    return feats.add_prefix(prefix)


def build_features(df: pd.DataFrame, featurizer):
    """Design matrix: one-hot (ligand, base, halogen) + descriptors (additive, aryl core).

    Returns (X, descriptor_columns); the second item lets the ridge model scale only
    the descriptor columns.
    """
    onehot = data.onehot(df, cols=("ligand_id", "base_id", "halogen"))
    add = attach(df, "additive", drop_constant(featurizer(sorted(df["additive"].unique()))), "add_")
    core = attach(df, "aryl_core", drop_constant(featurizer(sorted(df["aryl_core"].unique()))), "core_")
    X = pd.concat([onehot, add, core], axis=1)
    assert not X.isna().any().any()
    return X, list(add.columns) + list(core.columns)
