from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_environment(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    }


def _run(command: list[str], *, cwd: Path, home: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=_safe_environment(home),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _predictions(path: Path, expected_ids: pd.Series) -> pd.Series:
    frame = pd.read_csv(path, dtype={"row_id": str})
    if list(frame.columns) != ["row_id", "probability"]:
        raise ValueError("prediction header is invalid")
    if len(frame) != len(expected_ids) or frame["row_id"].duplicated().any():
        raise ValueError("prediction IDs are not unique and complete")
    if set(frame["row_id"]) != set(expected_ids.astype(str)):
        raise ValueError("prediction IDs do not match test.csv")
    values = pd.to_numeric(frame["probability"], errors="coerce")
    if not np.isfinite(values).all() or not values.between(1e-6, 1 - 1e-6).all():
        raise ValueError("prediction probabilities are invalid")
    return pd.Series(values.to_numpy(), index=frame["row_id"]).reindex(expected_ids)


def _verify_fixed_assets(app: Path, manifest: dict[str, str]) -> None:
    for relative, expected in manifest.items():
        path = app / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fixed asset is missing or unsafe: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"fixed asset changed: {relative}")


def check(app: Path, tests: Path) -> dict[str, object]:
    manifest = json.loads((tests / "fixed_assets.json").read_text(encoding="utf-8"))
    _verify_fixed_assets(app, manifest)

    for name in ("build.sh", "run.sh", "test.sh"):
        path = app / name
        if not path.is_file() or path.is_symlink() or not os.access(path, os.X_OK):
            raise ValueError(f"required executable is unavailable: {name}")

    with tempfile.TemporaryDirectory(prefix="mle-contract-") as temporary:
        work = Path(temporary)
        home = work / "home"
        home.mkdir()

        build = _run([str(app / "build.sh")], cwd=app, home=home, timeout=120)
        if build.returncode != 0:
            raise ValueError(f"build.sh failed: {build.stderr[-500:]}")

        tests_run = _run([str(app / "test.sh")], cwd=app, home=home, timeout=240)
        if tests_run.returncode != 0:
            raise ValueError(f"test.sh failed: {tests_run.stderr[-500:]}")

        train = app / "public/data/train.csv"
        validation = app / "public/data/validation.csv"
        test = app / "public/data/test.csv"
        expected_ids = pd.read_csv(test, dtype={"row_id": str})["row_id"]
        outputs = (work / "first.csv", work / "second.csv")
        for output in outputs:
            run = _run(
                [str(app / "run.sh"), str(train), str(validation), str(test), str(output)],
                cwd=app,
                home=home,
                timeout=45,
            )
            if run.returncode != 0:
                raise ValueError(f"run.sh failed: {run.stderr[-500:]}")
        first = _predictions(outputs[0], expected_ids)
        second = _predictions(outputs[1], expected_ids)
        maximum_delta = float(np.max(np.abs(first.to_numpy() - second.to_numpy())))
        if maximum_delta > 1e-12:
            raise ValueError(f"repeated predictions differ by {maximum_delta}")

        invalid_output = work / "invalid.csv"
        invalid = _run(
            [
                str(app / "run.sh"),
                str(work / "missing.csv"),
                str(validation),
                str(test),
                str(invalid_output),
            ],
            cwd=app,
            home=home,
            timeout=20,
        )
        if invalid.returncode == 0 or invalid_output.exists():
            raise ValueError("invalid input did not fail atomically")

    _verify_fixed_assets(app, manifest)
    return {"build": True, "tests": True, "run": True, "deterministic": True, "invalid_input": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.app.resolve(), args.tests.resolve()), sort_keys=True))
