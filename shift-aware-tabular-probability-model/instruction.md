# Improve our probability model for changing cohorts

We use the program in `/app` to estimate binary outcome probabilities for
future tabular cohorts. Its predictions are consumed by downstream systems that
need probabilities, not just class labels. I need the implementation improved
so it remains accurate and calibrated when future data differs from the older
labeled history.

Keep this command boundary working:

```sh
./run.sh TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV
```

`TRAIN_CSV` contains older labeled observations and `VALIDATION_CSV` contains a
newer labeled period. Both contain:

- `row_id`, an opaque unique identifier that must not be used as a feature;
- `event_time`, a numeric chronological coordinate;
- `segment`, an opaque categorical cohort value;
- opaque numeric feature columns, which may contain empty values; and
- `target`, the binary outcome `0` or `1`.

`TEST_CSV` represents a later period and has the same columns except `target`.
Write `OUTPUT_CSV` with exactly this header:

```csv
row_id,probability
```

The output must contain every test `row_id` exactly once, in any order, with no
extra rows or columns. Every probability must be finite and within the inclusive
range `0.000001` to `0.999999`.

Future cohorts can differ in feature scale, missingness, outlier frequency,
segment proportions, class balance, nonlinear structure, and the reliability
of relationships that looked useful in older data. Feature names, column order,
category tokens, and row order also vary. The same invocation may therefore
need to behave differently based on evidence in its supplied training and
validation periods. Do not assign meaning to identifier text, feature names, or
category spelling.

Probability quality is assessed with binary log loss on later outcomes. A
useful implementation must improve on a constant training-prevalence estimate
across the different cohort shifts rather than optimizing one representative
file. Repeated invocations with identical input must agree within `1e-12` per
row. A normal public-sized invocation must finish within 45 seconds using two
CPU threads.

Treat the supplied CSV fixtures as immutable examples. Other valid schemas,
row orders, seeds, and combinations of the documented shifts will be used. If
an input file is missing or structurally invalid, exit nonzero and do not leave
a completed output file.

You may replace or reorganize the modeling implementation while preserving the
command and CSV contracts. Leave the choice of preprocessing, model families,
training windows, ensembling, and calibration to the evidence in the data.

The following commands must work from `/app` without network access:

```sh
./build.sh
./test.sh
./run.sh public/data/train.csv public/data/validation.csv public/data/test.csv /tmp/predictions.csv
```

Add task-native tests for the behavior you implement, including representative
success, invalid-input, and edge cases.
