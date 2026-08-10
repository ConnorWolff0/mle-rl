#!/bin/sh
set -eu

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" - <<'PY'
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

root = Path.cwd()
sources = ("public/solution.py",)
for relative in sources:
    path = root / relative
    compile(path.read_text(), str(path), "exec")

with tempfile.TemporaryDirectory(prefix="mle-smoke-") as temporary:
    work = Path(temporary)
    train = root / "public/data/train.csv"
    validation = root / "public/data/validation.csv"
    test_path = root / "public/data/test.csv"
    output = work / "predictions.csv"
    subprocess.run(
        [
            str(root / "run.sh"),
            str(train),
            str(validation),
            str(test_path),
            str(output),
        ],
        check=True,
        env={
            **os.environ,
            "APP_DIR": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    test = pd.read_csv(test_path, dtype={"row_id": str})
    predictions = pd.read_csv(output, dtype={"row_id": str})
    assert list(predictions) == ["row_id", "probability"]
    assert len(predictions) == len(test)
    assert predictions["row_id"].is_unique
    assert set(predictions["row_id"]) == set(test["row_id"])
    assert predictions["probability"].between(1e-6, 1 - 1e-6).all()

    second = work / "predictions-second.csv"
    subprocess.run(
        [
            str(root / "run.sh"),
            str(train),
            str(validation),
            str(test_path),
            str(second),
        ],
        check=True,
        env={
            **os.environ,
            "APP_DIR": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    repeated = pd.read_csv(second, dtype={"row_id": str})
    joined = predictions.set_index("row_id").join(
        repeated.set_index("row_id"), lsuffix="_first", rsuffix="_second"
    )
    assert (
        joined["probability_first"] - joined["probability_second"]
    ).abs().max() <= 1e-12

    invalid_output = work / "invalid-output.csv"
    failed = subprocess.run(
        [
            str(root / "run.sh"),
            str(work / "missing.csv"),
            str(validation),
            str(test_path),
            str(invalid_output),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "APP_DIR": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert failed.returncode != 0
    assert not invalid_output.exists()

print("runtime smoke test: ok")
PY
