#!/usr/bin/env python3
"""Evaluate a solution on one seed from each scenario family."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import FAMILIES, make_scenario  # noqa: E402


PUBLIC_SEEDS = dict(zip(FAMILIES, (101, 202, 303, 404, 505)))


def log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(probability) + (1 - y) * np.log1p(-probability))))


def read_predictions(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"row_id": str})
    if list(frame.columns) != ["row_id", "probability"]:
        raise ValueError("header must be exactly row_id,probability")
    if len(frame) != len(ids) or frame["row_id"].duplicated().any():
        raise ValueError("each test ID must appear exactly once")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if set(frame["row_id"]) != set(ids.astype(str)):
        raise ValueError("row_id values do not match test.csv")
    if not np.isfinite(probability).all() or not (
        (probability >= 1e-6) & (probability <= 1 - 1e-6)
    ).all():
        raise ValueError("probabilities must be between 0.000001 and 0.999999")
    return pd.Series(probability.to_numpy(), index=frame["row_id"]).reindex(ids).to_numpy()


def evaluate(candidate: Path, family: str, seed: int) -> dict:
    train, validation, test, labels = make_scenario(family, seed)
    with tempfile.TemporaryDirectory(prefix="mle-public-") as temporary:
        work = Path(temporary)
        paths = [work / name for name in ("train.csv", "validation.csv", "test.csv")]
        for frame, path in zip((train, validation, test), paths):
            frame.to_csv(path, index=False)
        output = work / "predictions.csv"
        valid, error = True, None
        try:
            run = subprocess.run(
                [sys.executable, str(candidate), *map(str, paths), str(output)],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if run.returncode:
                raise ValueError(f"candidate exited {run.returncode}: {run.stderr[-200:].strip()}")
            probability = read_predictions(output, test["row_id"])
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            valid, error = False, str(exc)
            probability = np.full(len(test), 0.5)

    y = labels.set_index("row_id").loc[test["row_id"], "target"].to_numpy(dtype=int)
    prior = (train["target"].sum() + 1) / (len(train) + 2)
    loss = log_loss(y, probability)
    prior_loss = log_loss(y, np.full(len(y), prior))
    return {
        "family": family,
        "valid": valid,
        "log_loss": loss,
        "prior_log_loss": prior_loss,
        "skill": 1 - loss / prior_loss,
        **({"error": error} if error else {}),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python evaluate.py CANDIDATE.py")
    candidate = Path(sys.argv[1]).resolve()
    cases = [evaluate(candidate, family, PUBLIC_SEEDS[family]) for family in FAMILIES]
    result = {
        "raw_F": float(np.mean([case["skill"] for case in cases])),
        "valid_scenarios": sum(case["valid"] for case in cases),
        "total_scenarios": len(cases),
        "cases": cases,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
