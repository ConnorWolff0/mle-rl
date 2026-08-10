# Compact shift-aware MLE environment

This repository is the runnable prototype. A coding agent receives a starter
binary-classification program and improves its future probability predictions
under distribution shift. Prediction quality is the only scored objective;
the time limit is only a feasibility check.

## What the agent changes

The editable file is `public/solution.py`. For each scenario it is invoked as:

```sh
python solution.py TRAIN_CSV VALIDATION_CSV TEST_CSV OUTPUT_CSV
```

`train.csv` and `validation.csv` contain a binary `target`. `test.csv` is later
in time and omits the target. The solution must write exactly two columns:

```csv
row_id,probability
```

The complete task contract is in `public/instructions.md`. The controlled agent
instructions are in `ROLLOUT_PROMPT.md`.

## Run the prototype

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Five visible development scenarios.
python public/evaluate.py public/solution.py

# Ten private final scenarios and normalized reward.
python private/grade.py public/solution.py --bank final

# Fast end-to-end smoke check.
./test.sh
```

The three agent-visible example splits are checked in under `public/data/`.
Their labels are not. Evaluations generate fresh CSVs in a temporary directory
for every scenario. To generate another inspectable example yourself:

```sh
python simulator.py /tmp/mle-sample --family composite --seed 7
```

That command writes `train.csv`, `validation.csv`, `test.csv`, and presenter-only
`test_labels.csv` under `/tmp/mle-sample`.

## Scoring

For each scenario, binary log loss is compared with a constant model that uses
the smoothed training positive rate:

```text
skill  = 1 - candidate_log_loss / reference_log_loss
raw_F  = mean(skill across ten scenarios)
reward = clip(raw_F / F_golden, 0, 1)
```

An invalid or timed-out scenario receives `0.5` predictions and remains in the
average. `anchors.json` stores the frozen normalization value.

## Repository boundary

The runnable files here are intentionally small:

```text
mle-env/
├── README.md
├── ROLLOUT_PROMPT.md
├── requirements.txt
├── simulator.py
├── anchors.json
├── test.sh
├── public/
│   ├── instructions.md
│   ├── solution.py
│   ├── evaluate.py
│   └── data/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
└── private/
    └── grade.py
```

Presentation slides, frozen rollout evidence, presenter-only test labels, the
promoted golden source, and detailed run narratives live separately in
`../mle-env-docs/`. They are not needed to execute this repository.

This is a concise modeling and reward-differentiation prototype, not a
production-hardened agent sandbox. A real rollout runner must copy only its
explicit allowlist into an isolated agent workspace.
