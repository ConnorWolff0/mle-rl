# Agent Serving Policy Lab

A deterministic, CPU-only environment for designing an online serving policy
for multi-turn LLM agents.

The policy must coordinate four decisions:

1. route each ready model turn to a worker;
2. schedule the ready turns assigned to that worker;
3. keep or discard a reusable model adapter after a request; and
4. choose adapter replicas to evict when memory is full.

The simulator models heterogeneous workers, queues, end-to-end SLOs, multi-turn
KV reuse, tool-call gaps, adapter loads and peer copies, and finite HBM, host,
and adapter memory. It uses logical time, the Python 3.12 standard library, and
no GPU or network access.

## Quick start

```sh
cd environment
./build.sh
./test.sh
./run.sh scenarios/public.json /tmp/serving-report.json
python3 -m json.tool /tmp/serving-report.json
```

Edit [`environment/policy.py`](environment/policy.py). The callback and state
schemas are specified in [`environment/CONTRACT.md`](environment/CONTRACT.md),
and [`instruction.md`](instruction.md) contains the task prompt.

The included five-program scenario is a development fixture that exercises the
complete policy lifecycle. Its score is not a benchmark result.

## Evaluation assets

This public folder intentionally omits the private 1,024-program evaluation
bank, trusted grader, workload generator, rollout evidence, and tuned reference
policy. Keeping those assets private preserves the usefulness of the benchmark;
the public simulator, starter policy, fixture, and contract tests are sufficient
for development and integration.

This is a compact trace-driven simulator, not a hardware-fidelity model of a
specific GPU serving stack. It is intended to isolate the policy decisions and
make them fast, deterministic, and objectively scoreable on a CPU.
