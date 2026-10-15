"""Shared fixtures. Tests need neither network access nor RDKit."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORES = ["C1=CC=CC=C1", "C1=NC=CC=C1", "C1=CC=C(CC)C=C1", "C1=CC=C(OC)C=C1", "C1=CC=C(C(F)(F)F)C=C1"]
HALOGENS = ["Cl", "Br", "I"]


@pytest.fixture
def reactions():
    """Small full-factorial table shaped like the real one: 2 x 3 x 2 x (5 cores x 3 halogens) = 180 reactions."""
    rng = np.random.default_rng(0)
    rows = []
    for lig, add, base, (core, hal) in itertools.product(
        ["CCP(CC)CC", "CCCP(CCC)CCC"],
        ["CC1=CC=NO1", "CCC1=CC=NO1", "COC1=CC=NO1"],
        ["CN1CCCN2C1=NCCC2", "CCN=P(N(C)C)(N(C)C)N(C)C"],
        itertools.product(CORES, HALOGENS),
    ):
        rows.append({"ligand": lig, "additive": add, "base": base, "aryl_halide": hal + core,
                     "yield_pct": float(rng.uniform(0, 100))})
    return pd.DataFrame(rows)
