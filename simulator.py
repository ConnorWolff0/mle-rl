"""Generate chronological tabular classification scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAMILIES = (
    "linear_outliers",
    "nonlinear",
    "temporal_spurious",
    "heterogeneous",
    "composite",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def make_scenario(
    family: str,
    seed: int,
    n_train: int = 1200,
    n_validation: int = 400,
    n_test: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, test, and test-label tables."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")

    rng = np.random.default_rng(seed)
    k = 10
    perm = rng.permutation(k)
    scales = rng.choice([-1.0, 1.0], k) * 10.0 ** rng.uniform(-3.0, 3.0, k)
    offsets = rng.uniform(-2.0, 2.0, k) * np.abs(scales)
    names = [f"f_{rng.integers(0, 16**8):08x}" for _ in range(k)]
    segment_tokens = [f"g_{rng.integers(0, 16**6):06x}" for _ in range(3)]
    base_missing = rng.uniform(0.015, 0.07, k)
    law_scale = rng.uniform(0.88, 1.12, 6)
    intercept = rng.uniform(-0.25, 0.25)
    id_token = f"{rng.integers(0, 16**6):06x}"

    def rows(n: int, split: str) -> tuple[pd.DataFrame, np.ndarray]:
        phase = {"train": 0.0, "validation": 0.55, "test": 1.0}[split]
        lo, hi = {"train": (0.0, 0.58), "validation": (0.60, 0.76), "test": (0.78, 1.0)}[split]
        t = rng.uniform(lo, hi, n)

        if family == "heterogeneous":
            start, end = np.array((0.62, 0.29, 0.09)), np.array((0.20, 0.25, 0.55))
        elif family == "composite":
            start, end = np.array((0.58, 0.32, 0.10)), np.array((0.18, 0.30, 0.52))
        else:
            start, end = np.array((0.48, 0.34, 0.18)), np.array((0.34, 0.38, 0.28))
        probs = start * (1.0 - phase) + end * phase
        seg = rng.choice(3, size=n, p=probs)
        z = rng.normal(size=(n, 8))

        if family == "linear_outliers":
            z[:, 0] += 0.35 * phase
            z[:, 1] -= 0.20 * phase
        if family == "nonlinear":
            z[:, :4] *= 1.0 + 0.30 * phase
            z[:, 0] += 0.35 * phase
        if family == "composite":
            z[:, 0] += 0.45 * phase
            z[:, 1] -= 0.30 * phase
            z[:, 2:4] *= 1.0 + 0.25 * phase

        linear = (
            1.25 * law_scale[0] * z[:, 0]
            - 1.00 * law_scale[1] * z[:, 1]
            + 0.72 * law_scale[2] * z[:, 2]
            + 0.28 * (seg == 1)
            - 0.20 * (seg == 2)
        )
        nonlinear = (
            1.35 * law_scale[3] * np.sin(1.35 * z[:, 0])
            + 1.05 * law_scale[4] * z[:, 1] * z[:, 2]
            + 1.00 * (np.abs(z[:, 3]) < 0.65)
            - 0.70
        )
        local = np.choose(
            seg,
            (
                1.45 * z[:, 0] - 0.85 * z[:, 1],
                -1.15 * z[:, 0] + 1.10 * z[:, 2] * z[:, 3],
                1.30 * np.sin(1.5 * z[:, 1]) - 0.85 * (z[:, 2] > 0) + 0.8 * z[:, 4],
            ),
        )

        if family == "linear_outliers":
            logit = linear
        elif family == "nonlinear":
            logit = nonlinear + 0.20 * z[:, 4]
        elif family == "temporal_spurious":
            logit = linear + 0.35 * law_scale[5] * (t - 0.5)
        elif family == "heterogeneous":
            logit = local
        else:
            logit = 0.45 * linear + 0.55 * nonlinear + 0.50 * local
        logit = logit + intercept
        y = rng.binomial(1, _sigmoid(logit)).astype(int)

        # Ten observed numeric columns: five useful raw variables, correlated
        # copies, a time-varying nuisance, and pure noise.  In temporal families
        # the nuisance tracks z0 early and reverses sign toward deployment.
        rho = 1.25 - 2.50 * np.clip((t - 0.22) / 0.78, 0.0, 1.0)
        nuisance = rho * z[:, 0] + rng.normal(0.0, 0.38, n)
        if family not in ("temporal_spurious", "composite"):
            nuisance = 0.65 * z[:, 0] + rng.normal(0.0, 0.75, n)
        raw = np.column_stack(
            [z[:, :6], nuisance, 0.65 * z[:, 1] + rng.normal(0, 0.75, n), z[:, 6:8]]
        )

        if family in ("linear_outliers", "composite"):
            end_rate = 0.065 if family == "linear_outliers" else 0.045
            rate = 0.012 * (1.0 - phase) + end_rate * phase
            hit = rng.random((n, 2)) < rate
            raw[:, :2] += hit * rng.normal(0.0, 14.0, (n, 2))

        miss_prob = np.broadcast_to(base_missing, (n, k)).copy()
        if family == "composite":
            miss_prob[:, 0] += (0.08 + 0.12 * phase) * _sigmoid(1.4 * z[:, 0])
            miss_prob[:, 2] += (0.05 + 0.09 * phase) * _sigmoid(-1.2 * z[:, 2])
        missing = rng.random((n, k)) < np.clip(miss_prob, 0.0, 0.45)

        visible = raw[:, perm] * scales + offsets
        visible[missing[:, perm]] = np.nan
        frame = pd.DataFrame(visible, columns=names)
        prefix = {"train": "tr", "validation": "va", "test": "te"}[split]
        frame.insert(0, "segment", [segment_tokens[i] for i in seg])
        frame.insert(0, "event_time", t)
        frame.insert(0, "row_id", [f"{prefix}_{id_token}_{i:05d}" for i in range(n)])
        order = rng.permutation(n)
        return frame.iloc[order].reset_index(drop=True), y[order]

    train, y_train = rows(n_train, "train")
    validation, y_validation = rows(n_validation, "validation")
    test, y_test = rows(n_test, "test")
    train["target"] = y_train
    validation["target"] = y_validation
    labels = pd.DataFrame({"row_id": test["row_id"], "target": y_test})
    return train, validation, test, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic scenario.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--family", choices=FAMILIES, default="composite")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-validation", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=600)
    args = parser.parse_args()
    train, validation, test, labels = make_scenario(
        args.family, args.seed, args.n_train, args.n_validation, args.n_test
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    validation.to_csv(args.out_dir / "validation.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)
    labels.to_csv(args.out_dir / "test_labels.csv", index=False)
    print(f"wrote {len(train)} train, {len(validation)} validation, and {len(test)} test rows to {args.out_dir}")


if __name__ == "__main__":
    main()
