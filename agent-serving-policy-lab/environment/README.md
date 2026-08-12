# Agent Serving Policy Lab

This workspace is a deterministic CPU simulation of multi-turn agent inference. A replaceable policy in `policy.py` routes ready turns, orders worker queues, decides whether used model adapters remain resident, and prioritizes adapter eviction under capacity pressure. The simulator owns KV placement with a fixed LRU rule so every candidate faces the same cache mechanism.

Run the development loop with:

```sh
./build.sh
./test.sh
./run.sh scenarios/public.json /tmp/serving-report.json
python3 -m json.tool /tmp/serving-report.json
```

The five-program public scenario is a lifecycle fixture. It produces adapter cold loads, resident hits, a peer copy, capacity eviction, multi-turn KV reuse, and calls all four policy methods. Its score is not part of deployment service quality.

`CONTRACT.md` is authoritative for callback schemas, logical timing, fixed KV behavior, asynchronous adapter preparation, validation, resource limits, report fields, and service-quality arithmetic. Deployment traffic pools 1,024 equally weighted programs from the complete four-by-four client-demand and deployment-profile matrix described there.

Files under `serving_sim/` and `scenarios/` are byte-immutable. Policy execution receives no scenario path or future trace. It starts from a read-only `/app`, has private writable `HOME` and `TMPDIR`, and observes client state only through documented callback views.

The environment uses Python 3.12's standard library and requires no network.
