# Build an agent-serving policy

A client runs multi-turn LLM agents on a small inference cluster. Each program can pause for a tool call and later return with a reusable KV prefix. Programs can also require reusable model adapters. A worker may already hold the right prefix or adapter, may copy one from another worker, may load an adapter from storage, or may have to recompute an evicted prefix. These operations compete with queues, limited memory, different worker speeds, bursts, and end-to-end deadlines.

Improve `/app/policy.py` so as many complete programs as possible finish within their SLO. Your policy makes four decisions:

1. choose a worker for each ready model turn with `route`;
2. order that worker's ready queue with `schedule`;
3. keep or discard an adapter replica after use with `adapter_disposition`; and
4. rank adapter replicas for removal when a worker needs space with `evict_adapter`.

The simulator owns KV placement. It keeps unfinished KV prefixes in HBM, removes finished prefixes, and uses deterministic LRU pressure handling. Your routing decisions still determine whether a later turn gets a local HBM hit, restores from host, copies a remote prefix, or recomputes it.

The service objective is the raw pooled fraction of whole programs that meet their end-to-end SLO. Deployment traffic has 1,024 equally weighted programs: 64 programs in each of 16 configurations. The four client-demand families are `stationary`, `rotating`, `abrupt`, and `mixed`; each is crossed with `balanced`, `adapter_pressure`, `crossed`, and `burst_recovery` deployment profiles. Every additional on-time program raises service quality by exactly `1/1024`. There is no normalization, clipping, reference-policy comparison, or auxiliary-metric bonus.

Preserve the exact callback contract in `CONTRACT.md` and the headless `./run.sh SCENARIO_JSON OUTPUT_JSON` interface. Decisions must be deterministic and may use only the observations supplied to the callbacks plus the policy's own observation history. The policy architecture is open: you may replace the starter, split your implementation into support modules, and add focused tests using Python 3.12's standard library.

Files under `serving_sim/` and `scenarios/` are byte-immutable. Do not derive behavior from the public fixture's IDs, array order, event order, or constants. The fixture is a five-program development example that deliberately exercises adapter cold loads, reuse, a peer copy, capacity eviction, multi-turn KV reuse, and all four callbacks; its score is not part of deployment service quality.

Start with:

```sh
cd /app
./build.sh
./test.sh
./run.sh scenarios/public.json /tmp/serving-report.json
python3 -m json.tool /tmp/serving-report.json
```

The simulator is deterministic, CPU-only, self-contained, and independent of host wall-clock speed. See `CONTRACT.md` for the complete schemas, timing rules, limits, and report arithmetic.
