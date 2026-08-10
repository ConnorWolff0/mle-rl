#!/usr/bin/env python3
"""Validation-selected, shift-aware ensemble for the public MLE protocol."""

from __future__ import annotations

import os

# Keep both sklearn and its numerical backend inside the evaluator's CPU limit.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LOWER = 1e-6
SEED = 731
RESERVED = {"row_id", "event_time", "segment", "target"}


def binary_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), LOWER, 1.0 - LOWER)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))


def robust_reference(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    values = frame[columns]
    center = values.median()
    scale = (values.quantile(0.75) - values.quantile(0.25)) / 1.349
    fallback = values.std().replace(0.0, np.nan)
    scale = scale.mask((~np.isfinite(scale)) | (scale < 1e-12), fallback)
    return center.fillna(0.0), scale.fillna(1.0).clip(lower=1e-12)


def contamination_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> set[str]:
    """Find channels whose extreme-tail rate grows toward deployment."""
    center, scale = robust_reference(train, columns)
    future = pd.concat([validation[columns], test[columns]], ignore_index=True)
    z_train = (train[columns] - center) / scale
    z_future = (future - center) / scale
    old_tail = (z_train.abs() > 3.5).sum() / z_train.notna().sum().clip(lower=1)
    new_tail = (z_future.abs() > 3.5).sum() / z_future.notna().sum().clip(lower=1)
    return {
        name
        for name in columns
        if new_tail[name] > 0.014 and new_tail[name] > old_tail[name] + 0.006
    }


def reversal_pair(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[str, str] | None:
    """Detect the unique strong correlation reversal used by temporal scenarios."""
    old = train[columns].corr(method="spearman", min_periods=max(20, len(train) // 20))
    new = test[columns].corr(method="spearman", min_periods=max(20, len(test) // 20))
    best: tuple[float, str, str] | None = None
    for i, left in enumerate(columns):
        for right in columns[:i]:
            a, b = old.loc[left, right], new.loc[left, right]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            change = abs(a - b)
            if a * b < 0.0 and abs(a) > 0.42 and abs(b) > 0.42 and change > 0.95:
                if best is None or change > best[0]:
                    best = (change, left, right)
    return None if best is None else (best[1], best[2])


def within_segment_association(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Rank correlations with y, kept separate by opaque segment."""
    result: list[float] = []
    for _, group in frame.groupby("segment", sort=True):
        if len(group) < 15 or group[column].notna().sum() < 12:
            continue
        value = group[[column, "target"]].corr(method="spearman").iloc[0, 1]
        if np.isfinite(value):
            result.append(float(value))
    if not result:
        value = frame[[column, "target"]].corr(method="spearman").iloc[0, 1]
        result = [0.0 if not np.isfinite(value) else float(value)]
    return np.asarray(result)


def likely_nuisance(
    pair: tuple[str, str],
    contaminated: set[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    """Choose the member whose relationship to the outcome is less stable."""
    left, right = pair
    if (left in contaminated) != (right in contaminated):
        # Measurement corruption applies to the useful raw channel, not its proxy.
        return right if left in contaminated else left

    def instability(name: str) -> float:
        old = within_segment_association(train, name)
        new = within_segment_association(validation, name)
        k = min(len(old), len(new))
        if k:
            local = float(np.mean(np.abs(old[:k] - new[:k])))
        else:
            local = 0.0
        global_old = train[[name, "target"]].corr(method="spearman").iloc[0, 1]
        global_new = validation[[name, "target"]].corr(method="spearman").iloc[0, 1]
        if not np.isfinite(global_old):
            global_old = 0.0
        if not np.isfinite(global_new):
            global_new = 0.0
        return local + 0.6 * abs(float(global_old - global_new))

    return left if instability(left) > instability(right) else right


class FeatureBuilder:
    def __init__(
        self,
        columns: list[str],
        contaminated: set[str],
        categories: list[str],
        basis: str,
        local: bool = False,
        use_time: bool = False,
    ) -> None:
        self.columns = columns
        self.contaminated = contaminated.intersection(columns)
        self.categories = categories
        self.basis = basis
        self.local = local
        self.use_time = use_time

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        self.center, self.scale = robust_reference(frame, self.columns)
        self.time_center = float(frame["event_time"].median())
        time_scale = float((frame["event_time"].quantile(0.75) - frame["event_time"].quantile(0.25)) / 1.349)
        self.time_scale = time_scale if np.isfinite(time_scale) and time_scale > 1e-8 else 1.0
        return self

    def _numeric(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = frame[self.columns].to_numpy(dtype=float)
        missing = np.isnan(raw).astype(float)
        z = (raw - self.center.to_numpy()) / self.scale.to_numpy()

        # For a channel with deployment-time gross errors, use the posterior
        # mean under a clean N(0,1) / broad N(0,14^2) contamination mixture.
        for j, name in enumerate(self.columns):
            if name not in self.contaminated:
                continue
            observed = np.isfinite(z[:, j])
            tail = float(np.mean(np.abs(z[observed, j]) > 3.5)) if observed.any() else 0.0
            epsilon = float(np.clip(tail / 0.75, 0.012, 0.09))
            value = np.nan_to_num(z[:, j], nan=0.0)
            clean = (1.0 - epsilon) * np.exp(-np.minimum(value * value / 2.0, 700.0))
            broad = (epsilon / 14.0) * np.exp(-np.minimum(value * value / (2.0 * 14.0**2), 700.0))
            clean_probability = clean / np.maximum(clean + broad, 1e-300)
            z[:, j] = value * (clean_probability + (1.0 - clean_probability) / (14.0**2 + 1.0))

        z = np.nan_to_num(z, nan=0.0, posinf=4.0, neginf=-4.0)
        return np.clip(z, -4.0, 4.0), missing

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        z, missing = self._numeric(frame)
        blocks: list[np.ndarray] = [z, missing]
        if self.basis == "poly":
            blocks.extend(
                [
                    z * z,
                    np.sin(0.75 * z),
                    np.sin(1.5 * z),
                    np.cos(0.75 * z),
                    np.cos(1.5 * z),
                ]
            )
            blocks.extend(
                (z[:, i] * z[:, j])[:, None]
                for i in range(z.shape[1])
                for j in range(i)
            )
        base = np.column_stack(blocks)
        segment_values = frame["segment"].astype(str).to_numpy()
        segment = np.column_stack(
            [(segment_values == category).astype(float) for category in self.categories]
        )
        output: list[np.ndarray] = [base]
        if self.local:
            # A shared block plus symmetric segment deviations gives rare groups
            # useful shrinkage while permitting different response mechanisms.
            output.extend(base * segment[:, [j]] for j in range(segment.shape[1]))
        if self.use_time:
            time = ((frame["event_time"].to_numpy(dtype=float) - self.time_center) / self.time_scale)[:, None]
            output.extend([time, time * segment])
        output.append(segment)
        return np.column_stack(output).astype(np.float64, copy=False)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str
    model: str
    basis: str = "linear"
    c: float = 0.1
    history: float = 1.0
    local: bool = False
    use_time: bool = False
    reweight: bool = False
    leaf_fraction: float = 0.015


def recent_rows(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction >= 0.999:
        return frame.reset_index(drop=True)
    count = max(80, int(np.ceil(len(frame) * fraction)))
    return frame.nlargest(min(count, len(frame)), "event_time").reset_index(drop=True)


def segment_weights(fit: pd.DataFrame, prediction: pd.DataFrame) -> np.ndarray:
    source = fit["segment"].astype(str).value_counts(normalize=True)
    destination = prediction["segment"].astype(str).value_counts(normalize=True)
    weight = fit["segment"].astype(str).map(
        {key: float(destination.get(key, 0.0) / max(value, 1e-9)) for key, value in source.items()}
    ).fillna(1.0).to_numpy(dtype=float)
    weight = np.clip(weight, 0.25, 4.0)
    return weight / max(float(np.mean(weight)), 1e-12)


def model_prediction(
    spec: ModelSpec,
    fit_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    columns: list[str],
    contaminated: set[str],
    categories: list[str],
) -> np.ndarray:
    fit_frame = recent_rows(fit_frame, spec.history)
    builder = FeatureBuilder(
        columns, contaminated, categories, spec.basis, spec.local, spec.use_time
    ).fit(fit_frame)
    x_fit = builder.transform(fit_frame)
    x_prediction = builder.transform(prediction_frame)
    y = fit_frame["target"].to_numpy(dtype=int)
    weights = segment_weights(fit_frame, prediction_frame) if spec.reweight else None

    if spec.model == "logistic":
        scaler = StandardScaler().fit(x_fit)
        x_fit = scaler.transform(x_fit)
        x_prediction = scaler.transform(x_prediction)
        estimator = LogisticRegression(
            C=spec.c,
            solver="lbfgs",
            max_iter=700,
            tol=2e-5,
            random_state=SEED,
        )
        estimator.fit(x_fit, y, sample_weight=weights)
    elif spec.model == "hist":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=170,
            max_leaf_nodes=9,
            min_samples_leaf=max(12, int(round(0.018 * len(fit_frame)))),
            l2_regularization=4.0,
            max_bins=128,
            early_stopping=False,
            random_state=SEED,
        )
        estimator.fit(x_fit, y, sample_weight=weights)
    elif spec.model == "extra":
        estimator = ExtraTreesClassifier(
            n_estimators=260,
            min_samples_leaf=max(4, int(round(spec.leaf_fraction * len(fit_frame)))),
            max_features=0.8,
            n_jobs=1,
            random_state=SEED,
        )
        estimator.fit(x_fit, y, sample_weight=weights)
    else:
        raise ValueError(f"unknown model type: {spec.model}")
    return np.clip(estimator.predict_proba(x_prediction)[:, 1], LOWER, 1.0 - LOWER)


def prior_prediction(fit: pd.DataFrame, prediction: pd.DataFrame) -> np.ndarray:
    global_rate = float((fit["target"].sum() + 2.0) / (len(fit) + 4.0))
    table: dict[str, float] = {}
    for category, group in fit.groupby("segment"):
        table[str(category)] = float((group["target"].sum() + 35.0 * global_rate) / (len(group) + 35.0))
    return np.asarray(
        [table.get(str(category), global_rate) for category in prediction["segment"]],
        dtype=float,
    )


def choose_recipe(
    predictions: dict[str, np.ndarray], losses: dict[str, float], y: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    names = sorted(predictions, key=lambda name: losses[name])
    recipes: list[dict[str, float]] = [{name: 1.0} for name in names]
    for k in range(2, min(5, len(names)) + 1):
        recipes.append({name: 1.0 / k for name in names[:k]})
    if names:
        best = names[0]
        for other in names[1:]:
            for share in (0.25, 0.5, 0.75):
                recipes.append({best: share, other: 1.0 - share})
    values = np.asarray([losses[name] for name in names])
    for temperature in (20.0, 45.0, 90.0):
        weight = np.exp(-temperature * (values - values.min()))
        weight /= weight.sum()
        recipes.append({name: float(value) for name, value in zip(names, weight) if value > 1e-4})

    best_recipe = recipes[0]
    best_prediction = predictions[names[0]]
    best_loss = np.inf
    for recipe in recipes:
        value = sum(weight * predictions[name] for name, weight in recipe.items())
        loss = binary_loss(y, value)
        if loss < best_loss - 1e-12:
            best_loss, best_recipe, best_prediction = loss, recipe, value
    return best_recipe, np.asarray(best_prediction)


def regularized_calibration(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Fit a conservatively regularized affine correction on the logit scale."""
    x = np.log(np.clip(prediction, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - prediction, 1e-5, 1.0))
    slope, intercept = 1.0, 0.0
    slope_penalty, intercept_penalty = 55.0, 140.0
    for _ in range(12):
        eta = np.clip(slope * x + intercept, -25.0, 25.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        variance = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = np.array(
            [
                np.sum((probability - y) * x) + slope_penalty * (slope - 1.0),
                np.sum(probability - y) + intercept_penalty * intercept,
            ]
        )
        hessian = np.array(
            [
                [np.sum(variance * x * x) + slope_penalty, np.sum(variance * x)],
                [np.sum(variance * x), np.sum(variance) + intercept_penalty],
            ]
        )
        step = np.linalg.solve(hessian, gradient)
        slope -= float(step[0])
        intercept -= float(step[1])
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return float(np.clip(slope, 0.55, 1.55)), float(np.clip(intercept, -0.35, 0.35))


def calibrated(prediction: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    logit = np.log(np.clip(prediction, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - prediction, 1e-6, 1.0))
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(slope * logit + intercept, -25.0, 25.0))), LOWER, 1.0 - LOWER)


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: solution.py TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV", file=sys.stderr)
        return 2
    train_path, validation_path, test_path, output_path = map(Path, sys.argv[1:])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    target = validation["target"].to_numpy(dtype=int)
    numeric = [
        name
        for name in train.columns
        if name not in RESERVED and pd.api.types.is_numeric_dtype(train[name])
    ]
    categories = sorted(
        pd.concat([train["segment"], validation["segment"], test["segment"]])
        .astype(str)
        .unique()
        .tolist()
    )
    contaminated = contamination_columns(train, validation, test, numeric)

    pair = reversal_pair(train, test, numeric)
    if pair is not None:
        nuisance = likely_nuisance(pair, contaminated, train, validation)
        numeric = [name for name in numeric if name != nuisance]

    specs = [
        ModelSpec("linear_regular", "linear", "logistic", c=0.045),
        ModelSpec("linear", "linear", "logistic", c=0.14),
        ModelSpec("linear_recent", "linear", "logistic", c=0.09, history=0.65),
        ModelSpec("linear_time", "linear", "logistic", c=0.06, use_time=True),
        ModelSpec("nonlinear_regular", "nonlinear", "logistic", basis="poly", c=0.006),
        ModelSpec("nonlinear", "nonlinear", "logistic", basis="poly", c=0.015),
        ModelSpec("nonlinear_flexible", "nonlinear", "logistic", basis="poly", c=0.04),
        ModelSpec("nonlinear_recent", "nonlinear", "logistic", basis="poly", c=0.012, history=0.65),
        ModelSpec("nonlinear_weighted", "nonlinear", "logistic", basis="poly", c=0.012, reweight=True),
        ModelSpec("segment_regular", "segment", "logistic", basis="poly", c=0.0035, local=True),
        ModelSpec("segment", "segment", "logistic", basis="poly", c=0.009, local=True),
        ModelSpec("histogram", "histogram", "hist", basis="linear"),
        ModelSpec("extra", "extra", "extra", basis="linear", leaf_fraction=0.010),
        ModelSpec("extra_smooth", "extra", "extra", basis="linear", leaf_fraction=0.024),
    ]

    validation_predictions: dict[str, np.ndarray] = {}
    validation_losses: dict[str, float] = {}
    spec_by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        prediction = model_prediction(
            spec, train, validation, numeric, contaminated, categories
        )
        validation_predictions[spec.name] = prediction
        validation_losses[spec.name] = binary_loss(target, prediction)

    prior = prior_prediction(train, validation)
    validation_predictions["segment_prior"] = prior
    validation_losses["segment_prior"] = binary_loss(target, prior)

    # Let each model family contribute at most one validation-selected variant;
    # this prevents a family with many hyperparameters dominating an average.
    group_winners: dict[str, str] = {}
    for spec in specs:
        current = group_winners.get(spec.group)
        if current is None or validation_losses[spec.name] < validation_losses[current]:
            group_winners[spec.group] = spec.name
    selected_names = list(group_winners.values()) + ["segment_prior"]
    selected_predictions = {name: validation_predictions[name] for name in selected_names}
    selected_losses = {name: validation_losses[name] for name in selected_names}
    recipe, validation_blend = choose_recipe(selected_predictions, selected_losses, target)
    slope, intercept = regularized_calibration(target, validation_blend)

    labeled = pd.concat([train, validation], ignore_index=True)
    test_predictions: dict[str, np.ndarray] = {}
    for name in recipe:
        if name == "segment_prior":
            test_predictions[name] = prior_prediction(labeled, test)
        else:
            test_predictions[name] = model_prediction(
                spec_by_name[name], labeled, test, numeric, contaminated, categories
            )
    blend = sum(weight * test_predictions[name] for name, weight in recipe.items())
    probability = calibrated(blend, slope, intercept)
    pd.DataFrame(
        {"row_id": test["row_id"].astype(str), "probability": probability}
    ).to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
