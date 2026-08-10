from __future__ import annotations

import argparse
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

SELECTION_SEEDS = {
    "linear_outliers": (41821, 90733),
    "nonlinear": (52919, 101771),
    "temporal_spurious": (63029, 112909),
    "heterogeneous": (74143, 126227),
    "composite": (85361, 139901),
}

FINAL_SEEDS = {
    "linear_outliers": (156007, 230123),
    "nonlinear": (171161, 241117),
    "temporal_spurious": (186733, 252131),
    "heterogeneous": (199933, 263167),
    "composite": (214087, 274177),
}

BANKS = {"selection": SELECTION_SEEDS, "final": FINAL_SEEDS}


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))


def _read_predictions(path: Path, expected_ids: pd.Series) -> np.ndarray:
    if not path.is_file():
        raise ValueError("output file was not created")
    frame = pd.read_csv(path, dtype={"row_id": str})
    if list(frame.columns) != ["row_id", "probability"]:
        raise ValueError("header must be exactly row_id,probability")
    if len(frame) != len(expected_ids) or frame["row_id"].duplicated().any():
        raise ValueError("output must contain each test row exactly once")
    if set(frame["row_id"]) != set(expected_ids.astype(str)):
        raise ValueError("output row_id set does not match test.csv")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if not np.isfinite(probability).all() or not (
        (probability >= 1e-6) & (probability <= 1 - 1e-6)
    ).all():
        raise ValueError("probabilities must be between 0.000001 and 0.999999")
    return pd.Series(probability.to_numpy(), index=frame["row_id"]).reindex(expected_ids).to_numpy()


def _run_case(candidate: Path, family: str, seed: int, replica: int, timeout: int) -> dict:
    train, validation, test, labels = make_scenario(family, seed)
    with tempfile.TemporaryDirectory(prefix="mle-env-") as temporary:
        work = Path(temporary)
        train_path = work / "train.csv"
        validation_path = work / "validation.csv"
        test_path = work / "test.csv"
        output_path = work / "predictions.csv"
        train.to_csv(train_path, index=False)
        validation.to_csv(validation_path, index=False)
        test.to_csv(test_path, index=False)
        valid, error = True, None
        try:
            run = subprocess.run(
                [
                    sys.executable,
                    str(candidate),
                    str(train_path),
                    str(validation_path),
                    str(test_path),
                    str(output_path),
                ],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if run.returncode != 0:
                tail = (run.stderr or run.stdout)[-300:].strip()
                raise ValueError(f"candidate exited {run.returncode}: {tail}")
            probability = _read_predictions(output_path, test["row_id"])
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            valid, error = False, str(exc)
            probability = np.full(len(test), 0.5)

    y = labels.set_index("row_id").loc[test["row_id"], "target"].to_numpy(dtype=int)
    candidate_loss = _log_loss(y, probability)
    prior = (float(train["target"].sum()) + 1.0) / (len(train) + 2.0)
    prior_loss = _log_loss(y, np.full(len(y), prior))
    return {
        "scenario": f"{family}-{replica}",
        "family": family,
        "valid": valid,
        "skill": 1.0 - candidate_loss / prior_loss,
        "log_loss": candidate_loss,
        "prior_log_loss": prior_loss,
        **({"error": error} if error else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a candidate on 10 private shift scenarios.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bank", choices=tuple(BANKS), default="final")
    parser.add_argument("--timeout", type=int, default=45, help="seconds allowed per scenario")
    parser.add_argument("--anchors", type=Path, default=ROOT / "anchors.json")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    seeds = BANKS[args.bank]

    scenarios = [
        _run_case(candidate, family, seed, replica, args.timeout)
        for family in FAMILIES
        for replica, seed in enumerate(seeds[family], 1)
    ]
    families = {
        family: {
            "skill": float(np.mean([case["skill"] for case in scenarios if case["family"] == family])),
            "valid_scenarios": sum(case["valid"] for case in scenarios if case["family"] == family),
            "scenarios": len(seeds[family]),
        }
        for family in FAMILIES
    }
    raw_f = float(np.mean([case["skill"] for case in scenarios]))
    result = {
        "bank": args.bank,
        "raw_F": raw_f,
        "valid_scenarios": sum(case["valid"] for case in scenarios),
        "total_scenarios": len(scenarios),
        "families": families,
        "scenarios": scenarios,
    }

    if args.anchors.is_file():
        anchors = json.loads(args.anchors.read_text())
        if "F_golden" in anchors:
            golden = float(anchors["F_golden"])
            if golden <= 0:
                raise ValueError("anchors require F_golden > 0")
            result["reward"] = float(np.clip(raw_f / golden, 0.0, 1.0))
            result["normalization"] = {"F_golden": golden}
        else:
            start = float(anchors.get("F_start", anchors.get("start")))
            target = float(anchors.get("F_target", anchors.get("target")))
            if target <= start:
                raise ValueError("anchors require F_target > F_start")
            result["reward"] = float(np.clip((raw_f - start) / (target - start), 0.0, 1.0))
            result["normalization"] = {"F_start": start, "F_target": target}

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
