#!/usr/bin/env bash
# Reproduce everything from scratch (about 3 minutes on a small VM).
set -euo pipefail
cd "$(dirname "$0")"
python -I src/download_data.py
python -I src/evaluate.py
python -I src/offset_analysis.py
python -I src/importance.py
python -I src/figures.py
# optional: descriptor experiment (needs RDKit)
if python -c "import rdkit" 2>/dev/null; then
  python -I src/evaluate_descriptors.py
else
  echo "RDKit not installed: skipping src/evaluate_descriptors.py (pip install -r requirements-rdkit.txt)"
fi
# optional and slow (5-10 min): stability of the comparison, enable with RUN_ROBUSTNESS=1
if [ "${RUN_ROBUSTNESS:-0}" = "1" ] && python -c "import rdkit" 2>/dev/null; then
  python -I src/robustness.py
fi
