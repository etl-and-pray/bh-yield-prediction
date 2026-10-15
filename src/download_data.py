"""Fetch the Buchwald-Hartwig HTE table from rxn4chemistry/rxn_yields (MIT license).

The workbook is not committed to this repository; run this script once.
Requires git. It shallow-clones the source repository into a temporary directory
and copies the single file needed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

REPO = "https://github.com/rxn4chemistry/rxn_yields"
SRC = Path("data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx")


def main() -> None:
    if data.RAW.exists():
        print("already present:", data.RAW)
        return
    data.RAW.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True,
                       env={**__import__("os").environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        shutil.copy(Path(tmp) / SRC, data.RAW)
    print("saved", data.RAW, f"({data.RAW.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
