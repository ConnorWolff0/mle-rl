# Frozen controlled-rollout prompt

Your task name ends in a two-digit number `NN`. Your only allowed workspace is
`/tmp/mle-controlled-NN`. Do not inspect the parent repository, another `/tmp`
workspace, hidden grader files, previous results, or reference solutions.

Read every supplied file in that workspace. Then improve only
`public/solution.py` for the MLE task in `public/instructions.md`.

Make your own deterministic preprocessing, architecture, history-window,
ensembling, and probability-calibration decisions. Do not exploit the protocol,
row IDs, fixed public labels, or filename/path information. The solution must
finish within the documented per-scenario limit.

Every rollout has the same budget:

- one agent turn;
- edits to `public/solution.py` only;
- read-only inspection and syntax/compile checks are allowed;
- at most one full invocation of
  `/opt/anaconda3/bin/python3 public/evaluate.py public/solution.py`;
- no additional generated scenarios or direct use of `test_labels.csv`;
- no follow-up hints or hidden-score feedback.

Finish by reporting the workspace path, whether you changed the starter, the
public score if you ran it, and any known timeout or protocol failure. Stop after
that report.
