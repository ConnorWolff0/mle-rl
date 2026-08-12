# Agent Serving Policy Lab contract

This document defines the complete simulator boundary and every observation available to `policy.py`. Client workloads change values and mixtures within this contract; they do not introduce different decision rules.

## Running a scenario

Run one JSON scenario from `/app` with:

```sh
./run.sh SCENARIO_JSON OUTPUT_JSON
```

Both arguments are required. The input must exist and the output's parent directory must already exist. A valid run writes one deterministic JSON report to `OUTPUT_JSON` and exits zero. Usage errors, malformed scenarios, import failures, callback exceptions, and invalid decisions produce a concise error on standard error and a nonzero exit. Standard output is not a protocol surface.

Time uses non-negative integer microseconds, memory uses non-negative integer bytes, and rates use positive integer tokens or bytes per second. Simulator time is logical; host wall-clock speed never affects an SLO.

Files under `serving_sim/` and `scenarios/` are byte-immutable source assets. Generated `__pycache__/` directories and bytecode are excluded from this source commitment. A build or test command that changes any other entry in either tree makes the deliverable invalid. `CONTRACT.md` and `run.sh` are behavioral interfaces rather than byte-hashed assets. The deliverable is `policy.py`, optional support modules, and any tests you add.

Each deployment scenario creates a fresh policy process in isolated user, mount, PID, IPC, and network namespaces. `/app` is its read-only working directory. Its private `HOME` and `TMPDIR` are writable, while `/proc` and `/sys` are covered. Workload-construction inputs, future events, and orchestration state are unavailable. The policy receives client state only through the callback values below. It may use those observations, deterministic state retained from earlier callbacks, and static code or data shipped in `/app`; it must not use ambient time, process state, filesystem metadata, network data, or other side channels.

## Validation and resource limits

Before build, `/app` may contain at most 5,000 regular files and in-tree symlinks totaling at most 256 MiB of regular-file data. Symlink targets must be relative and resolve inside `/app`; special files are rejected, and `policy.py` must be a regular file.

The service checks execute `./build.sh` and then `./test.sh` from a writable `/app`. Each entrypoint must be executable and may be a symlink only when its target remains inside `/app`. Build has a 60-second wall limit and tests have a 90-second wall limit. Each command has 120 CPU seconds, 1.5 GiB of address space, a 4 MiB file/output limit, at most 64 open file descriptors and 64 processes, no core dumps, and no network. A command that fails, times out, or changes committed source assets invalidates the candidate.

The command-interface checks run `./run.sh scenarios/public.json OUTPUT_JSON` twice from a read-only `/app`, then run a different contract-valid scenario. Each run has a 60-second wall limit and the same resource limits as build and tests. Reports must match simulator behavior and replay byte-for-byte after JSON serialization. These runs establish command compatibility; their `score` values do not contribute to deployment service quality.

A policy process has 3 seconds to import and construct `Policy`, 2 seconds per callback, and 30 seconds for a complete scenario after startup. It has 10 CPU seconds, 1 GiB of address space, a 4 MiB file/output limit, at most 64 file descriptors and 64 processes, no core dumps, and at most 1,000,000 bytes per RPC observation or response. The outer task has the 2-CPU, 4-GiB limits in `task.toml`; these per-process limits are tighter. Any import, protocol, callback, serialization, resource, or isolation failure invalidates the deployment candidate, making its service quality zero.

## Scenario JSON

The top-level object contains `config`, `adapters`, `instances`, and `programs`:

```json
{
  "config": {
    "kv_bytes_per_token": 2048,
    "slo_us": 5000000,
    "hbm_to_host_bytes_per_second": 24000000000,
    "host_to_hbm_bytes_per_second": 24000000000,
    "peer_bytes_per_second": 50000000000,
    "transfer_fixed_us": 40,
    "adapter_peer_bytes_per_second": 16000000000,
    "adapter_transfer_fixed_us": 70
  },
  "adapters": [
    {
      "key": "adapter-opaque-17",
      "size_bytes": 536870912
    }
  ],
  "instances": [
    {
      "id": "worker-a",
      "hbm_capacity_bytes": 268435456,
      "host_capacity_bytes": 536870912,
      "prefill_tokens_per_second": 60000,
      "decode_tokens_per_second": 3000,
      "adapter_capacity_bytes": 1073741824,
      "adapter_load_bytes_per_second": 8000000000
    }
  ],
  "programs": [
    {
      "id": "program-opaque-23",
      "arrival_us": 0,
      "slo_us": 4800000,
      "adapter_key": "adapter-opaque-17",
      "turns": [
        {
          "new_input_tokens": 900,
          "output_tokens": 120,
          "output_token_hint": 128,
          "tool_gap_us": 350000,
          "tool_class": "shell",
          "tool_gap_hint_us": 300000
        }
      ]
    }
  ]
}
```

Validation rules:

- IDs and adapter keys are non-empty strings, unique within their respective collections, and must be treated as opaque.
- Capacities, rates, token counts, adapter sizes, `slo_us`, and `kv_bytes_per_token` are positive integers. `arrival_us`, `tool_gap_us`, `tool_gap_hint_us`, `host_capacity_bytes`, `adapter_capacity_bytes`, and fixed transfer times may be zero. Boolean values are not integers.
- `adapters` may be omitted and defaults to an empty array. Every `adapter_key` used by a program must name a declared adapter. Each used adapter must fit at least one worker's adapter capacity.
- Every program contains at least one turn. Its `slo_us` may be omitted to use `config.slo_us`. Its `adapter_key` may be omitted for a base-model program.
- `output_tokens` and `tool_gap_us` are realized trace outcomes. They are not included in policy observations before they occur.
- `output_token_hint` is the positive output-length estimate visible to the policy. It may be omitted in a development scenario, in which case the realized value is used. Client estimates may differ from realized output.
- `tool_class` is a non-empty opaque category and defaults to `generic`. `tool_gap_hint_us` is an optional non-negative duration estimate; it is neither an absolute return time nor a guarantee.
- A program's largest realized KV prefix must fit in at least one worker's HBM.
- Object-key order, array order, IDs, numeric values, and adapter popularity are not stable between scenarios.

Unknown object fields are ignored. Invalid types, duplicate identifiers, missing required fields, impossible cache or adapter sizes, negative quantities, and non-integral numbers are rejected.

## Policy module

`policy.py` must export a class named `Policy` that is constructible without arguments. One instance serves an entire scenario and may keep deterministic state between calls. The public API has exactly four callbacks: `route`, `schedule`, `adapter_disposition`, and `evict_adapter`. There is no candidate callback for KV placement or KV eviction.

The immutable view classes are exported from `serving_sim.models`. They are frozen dataclasses and use tuples rather than mutable simulator objects.

### Observable types

`RequestView` describes a model turn when it becomes ready:

```text
id: str
program_id: str
turn_index: int
ready_us: int
new_input_tokens: int
output_token_hint: int
previous_tokens: int
cache_bytes_estimate: int
deadline_us: int
adapter_key: str | None
adapter_size_bytes: int
```

`previous_tokens` is the accumulated KV prefix before this turn. `cache_bytes_estimate` estimates the post-turn KV size using `output_token_hint`. `deadline_us` is the program's absolute SLO deadline. `adapter_size_bytes` is zero when `adapter_key` is `None`.

`ToolView` describes the most recently completed model turn:

```text
tool_class: str
gap_hint_us: int | None
```

The gap hint is an estimated duration, not a future event timestamp.

`InstanceState` contains:

```text
id: str
estimated_available_us: int
running_request_id: str | None
queued_request_ids: tuple[str, ...]
hbm_used_bytes: int
hbm_capacity_bytes: int
host_used_bytes: int
host_capacity_bytes: int
prefill_tokens_per_second: int
decode_tokens_per_second: int
cache_copy_available_us: int
adapter_used_bytes: int
adapter_capacity_bytes: int
adapter_load_bytes_per_second: int
adapter_load_available_us: int
```

`estimated_available_us` uses public request information, including the running request's output-length hint. `cache_copy_available_us` and `adapter_load_available_us` report the next free time of the worker's two independent copy engines. Realized model completion remains unknown until it occurs.

`SessionState` contains:

```text
id: str
next_turn_index: int
completed_turns: int
cache_instance_id: str | None
cache_tier: "hbm" | "host" | None
cache_bytes: int
cached_tokens: int
last_access_us: int
finished: bool
tool: ToolView
deadline_us: int
cache_available_us: int
```

`cache_available_us` can be later than `now_us` while a simulator-owned KV copy is in flight.

`CacheEntryView` contains:

```text
session_id: str
instance_id: str
tier: "hbm" | "host"
size_bytes: int
tokens: int
last_access_us: int
completed_turns: int
finished: bool
tool: ToolView
available_us: int
```

`AdapterReplicaView` contains:

```text
adapter_key: str
instance_id: str
size_bytes: int
available_us: int
last_access_us: int
use_count: int
in_use: bool
```

A replica can be listed before its load or copy completes; `available_us` says when it becomes usable. `in_use` is true while model work or a peer copy pins it.

`ClusterState` contains:

```text
now_us: int
instances: tuple[InstanceState, ...]
sessions: tuple[SessionState, ...]
cache_entries: tuple[CacheEntryView, ...]
programs_completed: int
programs_arrived: int
kv_bytes_per_token: int
hbm_to_host_bytes_per_second: int
host_to_hbm_bytes_per_second: int
peer_bytes_per_second: int
transfer_fixed_us: int
adapter_replicas: tuple[AdapterReplicaView, ...]
adapter_peer_bytes_per_second: int
adapter_transfer_fixed_us: int
```

It also provides `state.instance(instance_id)`, `state.session(session_id)`, and `state.adapter_replicas_on(instance_id)`. The first two raise `KeyError` for unknown IDs. Collections may arrive in different deterministic orders; join values by opaque ID.

`sessions` contains only programs whose first arrival has occurred. `programs_arrived` is the count observed so far. The policy is never told future program IDs, future arrivals, future realized output lengths or tool gaps, or the eventual scenario size.

### `route`

```python
def route(self, request: RequestView, state: ClusterState) -> str:
    ...
```

This callback runs once when a turn becomes ready. Return exactly one declared worker ID. The request joins that worker's queue, and the simulator immediately begins or reserves the earliest feasible preparation of its required adapter. Routing also determines whether the existing KV prefix will be local, restored from host, copied from another worker, or recomputed after eviction.

### `schedule`

```python
def schedule(
    self,
    instance_id: str,
    queued_requests: tuple[RequestView, ...],
    state: ClusterState,
) -> Sequence[str]:
    ...
```

This callback runs whenever a worker can select queued work. Return an exact permutation of the supplied request IDs: no missing, unknown, or duplicate IDs. The first ID starts next. The remaining order is provisional because the callback runs again after state changes. An empty queue requires an empty result.

### `adapter_disposition`

```python
def adapter_disposition(
    self,
    replica: AdapterReplicaView,
    state: ClusterState,
) -> str:
    ...
```

This callback runs after each adapter-using request completes, if that exact replica is still present. Return exactly `"keep"` or `"evict"`. Keeping the replica makes later local use possible but consumes adapter capacity. Evicting it removes the reusable replica; a future request must load or copy it again. Base-model requests do not call this method.

### `evict_adapter`

```python
def evict_adapter(
    self,
    instance_id: str,
    required_bytes: int,
    candidates: tuple[AdapterReplicaView, ...],
    state: ClusterState,
) -> Sequence[str]:
    ...
```

This callback runs when a worker needs adapter capacity that cannot become available without removing reusable replicas. `required_bytes` is the positive deficit for the current decision. Candidates belong to `instance_id`, are available, and are eligible for removal. Return unique candidate adapter keys in victim-priority order. The simulator removes the smallest prefix of that order that makes the requested reservation feasible. Extra valid candidates may follow. Unknown keys, duplicates, keys from another worker, or an ordering that cannot free enough space are invalid.

## Simulation behavior

Programs arrive at `arrival_us` and their turns execute in order. After a model turn finishes, its realized `tool_gap_us` elapses before the next turn becomes ready. Every request is eventually served; the policy cannot drop work or alter tokens, adapters, arrivals, tool outcomes, hardware, or deadlines.

Each worker runs one model request at a time. Integer durations round up to the next microsecond:

```text
duration_us(units, rate) = ceil(units * 1,000,000 / rate)

prefill_us = duration_us(prefilled_tokens, prefill_tokens_per_second)
decode_us  = duration_us(realized_output_tokens, decode_tokens_per_second)

kv_copy_us = transfer_fixed_us + duration_us(kv_bytes, selected_kv_rate)
adapter_copy_us = adapter_transfer_fixed_us
                + duration_us(adapter_bytes, selected_adapter_rate)
```

For KV, the selected rate is host-to-HBM for a local host restore, peer bandwidth for remote HBM, `min(peer bandwidth, host-to-HBM bandwidth)` for remote host, and HBM-to-host for pressure demotion. A local HBM hit has no copy. Without a reusable prefix, all `previous_tokens` are recomputed at the destination's prefill rate. New input tokens are always prefetched.

Each worker has a KV copy engine and a separate adapter load/copy engine. A KV peer copy reserves both endpoint KV engines. A cold adapter load reserves the destination adapter engine; an adapter peer copy reserves both endpoint adapter engines. Adapter preparation can overlap queued or running model work, and multiple requests for the same pending local replica coalesce. The simulator chooses the earliest completion between a cold adapter load and an eligible peer copy, with a cold load winning an exact tie. A selected request's model compute waits until both its KV prerequisite and adapter prerequisite are ready, so overlapping prerequisite waits are combined with `max`, not added twice.

Realized output tokens determine decode duration and final KV size. Hints affect observations and estimated availability only. Transfers, capacity reservations, and model work advance logical time; host wall time has no role.

### Fixed KV policy

KV placement is the same for every candidate:

- an unfinished program's prefix stays in HBM after a turn;
- a completed program's prefix is removed immediately;
- HBM pressure considers eligible prefixes in least-recently-used order, demoting each to local host when feasible and otherwise evicting it;
- host pressure evicts eligible prefixes in least-recently-used order; and
- active requests and in-flight copies remain protected until safe to move or remove.

A returning turn can therefore observe a local HBM hit, a local host restore, a remote HBM or host copy, or a miss that recomputes the accumulated prefix. Routing and queue timing influence these outcomes even though KV eviction order is fixed.

Equal-time events and remaining ties use stable opaque-ID ordering, so identical inputs and policy history replay exactly.

## Report and service quality

The output is a JSON object:

```json
{
  "score": 0.73,
  "metrics": {
    "programs_total": 100,
    "programs_completed": 100,
    "programs_within_slo": 73
  },
  "programs": [],
  "requests": []
}
```

Each program trace reports its ID, arrival, completion, end-to-end latency, applied SLO, `within_slo`, and request IDs. Each request trace reports routing, ready/start/finish times, queue wait, service and transfer time, KV source, reused and recomputed tokens, transfer bytes, realized tokens, adapter key and source, adapter wait, and adapter transfer bytes. `metrics` also contains deterministic latency, cache, adapter, transfer, recomputation, and goodput diagnostics. Diagnostics explain behavior but do not alter service quality.

For program `p`:

```text
latency_p = final_completion_us - arrival_us
within_slo_p = latency_p <= program_slo_us
```

Tool-call time remains part of end-to-end latency. A scenario report's `score` is its raw whole-program SLO attainment:

```text
score = programs_within_slo / programs_total
```

Deployment traffic is the complete cross of four client-demand families and four deployment profiles:

| Client-demand family | Meaning |
| --- | --- |
| `stationary` | adapter popularity remains comparatively stable |
| `rotating` | the popular adapter set changes by phase |
| `abrupt` | demand shifts sharply between adapter sets |
| `mixed` | stable, rotating, and shifting demand coexist |

| Deployment profile | Meaning |
| --- | --- |
| `balanced` | generally balanced workers and memory pressure |
| `adapter_pressure` | phase working sets exceed a worker's local adapter capacity |
| `crossed` | workers have different compute and adapter-loading strengths |
| `burst_recovery` | arrival bursts alternate with recovery intervals |

Each of the 16 configurations contains 64 programs, giving 1,024 equally weighted programs. Counts are pooled before division:

```text
service_quality = sum(programs_within_slo across 16 configurations) / 1024
```

Every program contributes exactly `1/1024`; equivalently, every 64-program configuration contributes 6.25% of service quality. The result is the unclipped `[0, 1]` fraction. No reference policy, case weighting, action cost, latency shaping, or diagnostic metric enters the calculation.
