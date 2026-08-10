#!/bin/sh
set -eu

export PYTHONDONTWRITEBYTECODE=1

python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

expected = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.1",
    "joblib": "1.4.2",
    "threadpoolctl": "3.6.0",
}
for package, wanted in expected.items():
    assert version(package) == wanted, (package, version(package), wanted)

for relative in ("public/solution.py",):
    path = Path(relative)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY

python3 -m pip check
