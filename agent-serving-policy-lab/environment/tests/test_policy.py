from __future__ import annotations

import unittest
from pathlib import Path

from policy import Policy
from serving_sim import load_workload, run_simulation


SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "public.json"
CALLBACKS = (
    "route",
    "schedule",
    "adapter_disposition",
    "evict_adapter",
)


class RecordingPolicy:
    def __init__(self) -> None:
        self.delegate = Policy()
        self.calls = {name: 0 for name in CALLBACKS}
        for callback in CALLBACKS:
            if not callable(getattr(self.delegate, callback, None)):
                raise TypeError(f"Policy.{callback} must be callable")

    def route(self, request, state):
        self.calls["route"] += 1
        return self.delegate.route(request, state)

    def schedule(self, instance_id, queued_requests, state):
        self.calls["schedule"] += 1
        return self.delegate.schedule(instance_id, queued_requests, state)

    def adapter_disposition(self, replica, state):
        self.calls["adapter_disposition"] += 1
        return self.delegate.adapter_disposition(replica, state)

    def evict_adapter(self, instance_id, required_bytes, candidates, state):
        self.calls["evict_adapter"] += 1
        return self.delegate.evict_adapter(
            instance_id,
            required_bytes,
            candidates,
            state,
        )


def simulate(policy_type=Policy) -> tuple[dict[str, object], dict[str, int]]:
    policy = RecordingPolicy()
    policy.delegate = policy_type()
    result = run_simulation(load_workload(SCENARIO), policy)
    return result.to_dict(), dict(policy.calls)


class LifecycleProbePolicy:
    """Deterministic fixture probe, separate from the editable candidate."""

    def route(self, request, state):
        session = state.session(request.program_id)
        if session.cache_instance_id is not None:
            return session.cache_instance_id
        return min(
            state.instances,
            key=lambda instance: (
                len(instance.queued_request_ids),
                max(instance.estimated_available_us, state.now_us),
                instance.id,
            ),
        ).id

    def schedule(self, instance_id, queued_requests, state):
        return [request.id for request in queued_requests]

    def adapter_disposition(self, replica, state):
        return "keep"

    def evict_adapter(self, instance_id, required_bytes, candidates, state):
        return [
            replica.adapter_key
            for replica in sorted(
                candidates,
                key=lambda replica: (
                    replica.last_access_us,
                    replica.use_count,
                    replica.adapter_key,
                ),
            )
        ]


class PolicyContractTests(unittest.TestCase):
    def test_public_workload_completes_and_replays_exactly(self) -> None:
        first_report, first_calls = simulate()
        second_report, second_calls = simulate()

        self.assertEqual(first_report, second_report)
        self.assertEqual(first_calls, second_calls)

        metrics = first_report["metrics"]
        programs = first_report["programs"]
        requests = first_report["requests"]
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(programs, list)
        self.assertIsInstance(requests, list)

        total = metrics["programs_total"]
        completed = metrics["programs_completed"]
        within_slo = metrics["programs_within_slo"]
        self.assertEqual(completed, total)
        self.assertEqual(len(programs), total)
        self.assertEqual(first_report["score"], within_slo / total)

        request_count = len(requests)
        self.assertEqual(total, 5)
        self.assertEqual(request_count, 7)
        self.assertEqual(first_calls["route"], request_count)
        self.assertEqual(first_calls["schedule"], request_count)
        self.assertEqual(first_calls["adapter_disposition"], request_count)
        self.assertEqual(metrics["adapter_requests"], request_count)
        self.assertEqual(metrics["finished_cache_evictions"], total)

    def test_fixture_causally_exercises_the_documented_lifecycle(self) -> None:
        report, calls = simulate(LifecycleProbePolicy)
        metrics = report["metrics"]
        requests = report["requests"]

        self.assertEqual(report["score"], 4 / 5)
        self.assertEqual(calls["route"], 7)
        self.assertEqual(calls["schedule"], 7)
        self.assertEqual(calls["adapter_disposition"], 7)
        self.assertEqual(calls["evict_adapter"], 2)
        self.assertEqual(metrics["adapter_cold_loads"], 3)
        self.assertEqual(metrics["adapter_resident_hits"], 3)
        self.assertEqual(metrics["adapter_peer_copies"], 1)
        self.assertEqual(metrics["adapter_evictions"], 2)
        self.assertEqual(metrics["cache_hbm_hits"], 2)

        sources = [request["adapter_source"] for request in requests]
        self.assertEqual(sources.count("cold"), 3)
        self.assertEqual(sources.count("resident"), 3)
        self.assertEqual(sum(source.startswith("peer:") for source in sources), 1)


if __name__ == "__main__":
    unittest.main()
