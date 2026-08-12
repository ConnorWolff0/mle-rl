from __future__ import annotations

import copy
import unittest
from typing import Any

from serving_sim.io import workload_from_dict
from serving_sim.models import ScenarioError


def scenario() -> dict[str, Any]:
    return {
        "config": {
            "kv_bytes_per_token": 2,
            "slo_us": 1_000_000,
            "hbm_to_host_bytes_per_second": 1_000_000,
            "host_to_hbm_bytes_per_second": 1_000_000,
            "peer_bytes_per_second": 2_000_000,
            "transfer_fixed_us": 5,
            "adapter_peer_bytes_per_second": 4_000_000,
            "adapter_transfer_fixed_us": 9,
        },
        "adapters": [{"key": "adapter-a", "size_bytes": 1_000}],
        "instances": [
            {
                "id": "worker-a",
                "hbm_capacity_bytes": 10_000,
                "host_capacity_bytes": 20_000,
                "prefill_tokens_per_second": 50_000,
                "decode_tokens_per_second": 2_000,
                "adapter_capacity_bytes": 2_000,
                "adapter_load_bytes_per_second": 3_000_000,
            }
        ],
        "programs": [
            {
                "id": "program-a",
                "arrival_us": 0,
                "adapter_key": "adapter-a",
                "turns": [
                    {
                        "new_input_tokens": 100,
                        "output_tokens": 10,
                        "output_token_hint": 12,
                        "tool_gap_us": 20_000,
                        "tool_class": "shell",
                        "tool_gap_hint_us": 25_000,
                    }
                ],
            }
        ],
    }


class ScenarioValidationTests(unittest.TestCase):
    def test_valid_scenario_loads_hints(self) -> None:
        workload = workload_from_dict(scenario())
        turn = workload.programs[0].turns[0]
        self.assertEqual(turn.output_token_hint, 12)
        self.assertEqual(turn.tool_class, "shell")
        self.assertEqual(turn.tool_gap_hint_us, 25_000)
        self.assertEqual(workload.programs[0].adapter_key, "adapter-a")
        self.assertEqual(workload.adapters[0].size_bytes, 1_000)
        self.assertEqual(workload.instances[0].adapter_capacity_bytes, 2_000)

    def test_boolean_is_not_accepted_as_an_integer(self) -> None:
        raw = copy.deepcopy(scenario())
        raw["config"]["kv_bytes_per_token"] = True
        with self.assertRaises(ScenarioError):
            workload_from_dict(raw)

    def test_duplicate_program_ids_are_rejected(self) -> None:
        raw = copy.deepcopy(scenario())
        raw["programs"].append(copy.deepcopy(raw["programs"][0]))
        with self.assertRaises(ScenarioError):
            workload_from_dict(raw)

    def test_cache_larger_than_every_instance_is_rejected(self) -> None:
        raw = copy.deepcopy(scenario())
        raw["instances"][0]["hbm_capacity_bytes"] = 100
        with self.assertRaises(ScenarioError):
            workload_from_dict(raw)

    def test_unknown_adapter_is_rejected(self) -> None:
        raw = copy.deepcopy(scenario())
        raw["programs"][0]["adapter_key"] = "missing"
        with self.assertRaises(ScenarioError):
            workload_from_dict(raw)

    def test_adapter_larger_than_every_instance_is_rejected(self) -> None:
        raw = copy.deepcopy(scenario())
        raw["instances"][0]["adapter_capacity_bytes"] = 999
        with self.assertRaises(ScenarioError):
            workload_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
