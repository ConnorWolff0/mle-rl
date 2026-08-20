Please improve `/app/policy.py` so more complete multi-turn programs finish within their end-to-end SLOs.

Programs may pause for tool calls and return with reusable KV prefixes. Some also need model adapters. Workers have different speeds and memory limits, so a turn may reuse local state, copy it from another worker, restore it from host memory, or recompute it after eviction.

The policy has four callbacks. `route` chooses a worker, `schedule` orders that worker's queue, `adapter_disposition` decides whether a used adapter stays resident, and `evict_adapter` ranks adapter replicas to remove when space is needed. The simulator handles KV placement with fixed LRU rules, but routing and scheduling still determine how much state can be reused.

Service quality is the share of complete programs that meet their SLO. It is measured across 1,024 equally weighted programs. The five-program public scenario is only a development fixture and is not part of that result.

Keep the callback interface in `CONTRACT.md` and the `./run.sh SCENARIO_JSON OUTPUT_JSON` command working. Decisions must be deterministic and may use only the supplied callback observations and state saved from earlier callbacks. You may replace the starter policy, add helper modules, and add tests using Python 3.12's standard library.

Leave files under `serving_sim/` and `scenarios/` unchanged. Do not special-case the public fixture's IDs, ordering, event sequence, or values.

You can run the development loop with:

```sh
cd /app
./build.sh
./test.sh
./run.sh scenarios/public.json /tmp/serving-report.json
python3 -m json.tool /tmp/serving-report.json
```

`CONTRACT.md` contains the exact callback schemas, timing rules, limits, and report format.
