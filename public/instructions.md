# Build a shift-aware probability model

Improve `solution.py`.  The evaluator repeatedly invokes:

```sh
python solution.py TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV
```

`TRAIN_CSV` contains older labeled rows. `VALIDATION_CSV` contains newer
labeled rows from the same world. Both contain:

- `row_id`: an opaque identifier; never use it as a feature;
- `event_time`: a numeric time coordinate;
- `segment`: an opaque categorical value;
- several opaque numeric feature columns; and
- `target`: the binary label, `0` or `1`.

`TEST_CSV` is later in time and has the same columns except `target`. Row
order, feature order,
feature names, numeric scales, category tokens, missingness, and the future
data distribution vary by scenario.

Write `OUTPUT_CSV` with exactly one row per test ID:

```csv
row_id,probability
```

Probabilities must be finite and between `0.000001` and `0.999999`.
If a scenario crashes, times out, or produces an invalid file, that scenario is
scored as the uninformative prediction `0.5`; it is never removed from the
average.

The score is future-cohort binary log-loss skill relative to a constant model
that predicts the smoothed training positive rate.  Scenarios include stable
linear structure, nonlinear geometry, temporal drift and spurious features,
changing segment mixtures, and combinations of those mechanisms.  The score
contains no runtime, model-count, or library penalty; the limits are simply 2
CPU threads, 2 GiB RAM, and 45 seconds per scenario.

Use validation to decide preprocessing, model family, recency window,
ensembling, and probability calibration. You may then refit using all labeled
rows. Fix all random seeds. Do not infer
meaning from column names, row IDs, or category spellings.

Try the starter on the five public scenarios:

```sh
python evaluate.py solution.py
```
