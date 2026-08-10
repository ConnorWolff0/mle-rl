#!/bin/sh
set -eu

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" - <<'PY'
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

root = Path.cwd()
sources = (
    "simulator.py",
    "public/evaluate.py",
    "public/solution.py",
    "private/grade.py",
)
for relative in sources:
    path = root / relative
    compile(path.read_text(), str(path), "exec")

anchors = json.loads((root / "anchors.json").read_text())
assert float(anchors["F_golden"]) > 0

with tempfile.TemporaryDirectory(prefix="mle-smoke-") as temporary:
    work = Path(temporary)
    subprocess.run(
        [
            sys.executable,
            str(root / "simulator.py"),
            str(work),
            "--family",
            "composite",
            "--seed",
            "7",
            "--n-train",
            "160",
            "--n-validation",
            "60",
            "--n-test",
            "80",
        ],
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = work / "predictions.csv"
    subprocess.run(
        [
            sys.executable,
            str(root / "public/solution.py"),
            str(work / "train.csv"),
            str(work / "validation.csv"),
            str(work / "test.csv"),
            str(output),
        ],
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    test = pd.read_csv(work / "test.csv", dtype={"row_id": str})
    predictions = pd.read_csv(output, dtype={"row_id": str})
    assert list(predictions) == ["row_id", "probability"]
    assert len(predictions) == len(test)
    assert predictions["row_id"].is_unique
    assert set(predictions["row_id"]) == set(test["row_id"])
    assert predictions["probability"].between(1e-6, 1 - 1e-6).all()

print("runtime smoke test: ok")
PY
