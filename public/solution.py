#!/usr/bin/env python3
"""Train a regularized logistic probability model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOWER = 1e-6


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: solution.py TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV",
            file=sys.stderr,
        )
        return 2
    train_path, validation_path, test_path, output_path = map(Path, sys.argv[1:])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    labeled = pd.concat([train, validation], ignore_index=True)
    numeric = [
        name for name in labeled.columns
        if name not in {"row_id", "segment", "target"}
    ]
    features = [*numeric, "segment"]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("segment", OneHotEncoder(handle_unknown="ignore"), ["segment"]),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("model", LogisticRegression(C=0.5, max_iter=500, random_state=7)),
    ])
    model.fit(labeled[features], labeled["target"].astype(int))
    probability = np.clip(model.predict_proba(test[features])[:, 1], LOWER, 1 - LOWER)
    pd.DataFrame({
        "row_id": test["row_id"].astype(str),
        "probability": probability,
    }).to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
