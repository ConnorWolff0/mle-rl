from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AdapterConfig,
    InstanceConfig,
    Program,
    ScenarioError,
    SimulationConfig,
    Turn,
    Workload,
)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioError(f"{name} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ScenarioError(f"{name} must be {qualifier}")
    return value


def workload_from_dict(raw: dict[str, Any]) -> Workload:
    if not isinstance(raw, dict):
        raise ScenarioError("scenario must be a JSON object")
    try:
        raw_config = raw["config"]
        raw_instances = raw["instances"]
        raw_programs = raw["programs"]
    except KeyError as error:
        raise ScenarioError(f"scenario is missing {error.args[0]!r}") from error

    if not isinstance(raw_config, dict):
        raise ScenarioError("config must be an object")
    config = SimulationConfig(
        kv_bytes_per_token=_positive_int(raw_config.get("kv_bytes_per_token"), "kv_bytes_per_token"),
        slo_us=_positive_int(raw_config.get("slo_us"), "slo_us"),
        hbm_to_host_bytes_per_second=_positive_int(
            raw_config.get("hbm_to_host_bytes_per_second"),
            "hbm_to_host_bytes_per_second",
        ),
        host_to_hbm_bytes_per_second=_positive_int(
            raw_config.get("host_to_hbm_bytes_per_second"),
            "host_to_hbm_bytes_per_second",
        ),
        peer_bytes_per_second=_positive_int(
            raw_config.get("peer_bytes_per_second"), "peer_bytes_per_second"
        ),
        transfer_fixed_us=_positive_int(
            raw_config.get("transfer_fixed_us", 0), "transfer_fixed_us", allow_zero=True
        ),
        adapter_peer_bytes_per_second=_positive_int(
            raw_config.get("adapter_peer_bytes_per_second", 1),
            "adapter_peer_bytes_per_second",
        ),
        adapter_transfer_fixed_us=_positive_int(
            raw_config.get("adapter_transfer_fixed_us", 0),
            "adapter_transfer_fixed_us",
            allow_zero=True,
        ),
    )

    if not isinstance(raw_instances, list) or not raw_instances:
        raise ScenarioError("instances must be a non-empty array")
    instances: list[InstanceConfig] = []
    instance_ids: set[str] = set()
    for index, item in enumerate(raw_instances):
        if not isinstance(item, dict):
            raise ScenarioError(f"instances[{index}] must be an object")
        instance_id = item.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ScenarioError(f"instances[{index}].id must be a non-empty string")
        if instance_id in instance_ids:
            raise ScenarioError(f"duplicate instance id {instance_id!r}")
        instance_ids.add(instance_id)
        instances.append(
            InstanceConfig(
                id=instance_id,
                hbm_capacity_bytes=_positive_int(
                    item.get("hbm_capacity_bytes"), f"instances[{index}].hbm_capacity_bytes"
                ),
                host_capacity_bytes=_positive_int(
                    item.get("host_capacity_bytes"),
                    f"instances[{index}].host_capacity_bytes",
                    allow_zero=True,
                ),
                prefill_tokens_per_second=_positive_int(
                    item.get("prefill_tokens_per_second"),
                    f"instances[{index}].prefill_tokens_per_second",
                ),
                decode_tokens_per_second=_positive_int(
                    item.get("decode_tokens_per_second"),
                    f"instances[{index}].decode_tokens_per_second",
                ),
                adapter_capacity_bytes=_positive_int(
                    item.get("adapter_capacity_bytes", 0),
                    f"instances[{index}].adapter_capacity_bytes",
                    allow_zero=True,
                ),
                adapter_load_bytes_per_second=_positive_int(
                    item.get("adapter_load_bytes_per_second", 1),
                    f"instances[{index}].adapter_load_bytes_per_second",
                ),
            )
        )

    raw_adapters = raw.get("adapters", [])
    if not isinstance(raw_adapters, list):
        raise ScenarioError("adapters must be an array")
    adapters: list[AdapterConfig] = []
    adapter_keys: set[str] = set()
    for index, item in enumerate(raw_adapters):
        if not isinstance(item, dict):
            raise ScenarioError(f"adapters[{index}] must be an object")
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise ScenarioError(f"adapters[{index}].key must be a non-empty string")
        if key in adapter_keys:
            raise ScenarioError(f"duplicate adapter key {key!r}")
        adapter_keys.add(key)
        adapters.append(
            AdapterConfig(
                key=key,
                size_bytes=_positive_int(
                    item.get("size_bytes"), f"adapters[{index}].size_bytes"
                ),
            )
        )

    if not isinstance(raw_programs, list) or not raw_programs:
        raise ScenarioError("programs must be a non-empty array")
    programs: list[Program] = []
    program_ids: set[str] = set()
    for index, item in enumerate(raw_programs):
        if not isinstance(item, dict):
            raise ScenarioError(f"programs[{index}] must be an object")
        program_id = item.get("id")
        if not isinstance(program_id, str) or not program_id:
            raise ScenarioError(f"programs[{index}].id must be a non-empty string")
        if program_id in program_ids:
            raise ScenarioError(f"duplicate program id {program_id!r}")
        program_ids.add(program_id)
        arrival_us = _positive_int(
            item.get("arrival_us", 0), f"programs[{index}].arrival_us", allow_zero=True
        )
        raw_turns = item.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ScenarioError(f"programs[{index}].turns must be a non-empty array")
        turns: list[Turn] = []
        for turn_index, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, dict):
                raise ScenarioError(
                    f"programs[{index}].turns[{turn_index}] must be an object"
                )
            tool_class = raw_turn.get("tool_class", "generic")
            if not isinstance(tool_class, str) or not tool_class:
                raise ScenarioError(
                    f"programs[{index}].turns[{turn_index}].tool_class "
                    "must be a non-empty string"
                )
            turns.append(
                Turn(
                    new_input_tokens=_positive_int(
                        raw_turn.get("new_input_tokens"),
                        f"programs[{index}].turns[{turn_index}].new_input_tokens",
                    ),
                    output_tokens=_positive_int(
                        raw_turn.get("output_tokens"),
                        f"programs[{index}].turns[{turn_index}].output_tokens",
                    ),
                    tool_gap_us=_positive_int(
                        raw_turn.get("tool_gap_us", 0),
                        f"programs[{index}].turns[{turn_index}].tool_gap_us",
                        allow_zero=True,
                    ),
                    output_token_hint=_positive_int(
                        raw_turn.get("output_token_hint", raw_turn.get("output_tokens")),
                        f"programs[{index}].turns[{turn_index}].output_token_hint",
                    ),
                    tool_class=tool_class,
                    tool_gap_hint_us=(
                        None
                        if raw_turn.get("tool_gap_hint_us") is None
                        else _positive_int(
                            raw_turn.get("tool_gap_hint_us"),
                            f"programs[{index}].turns[{turn_index}].tool_gap_hint_us",
                            allow_zero=True,
                        )
                    ),
                )
            )
        raw_slo = item.get("slo_us")
        slo_us = None if raw_slo is None else _positive_int(raw_slo, f"programs[{index}].slo_us")
        adapter_key = item.get("adapter_key")
        if adapter_key is not None and (
            not isinstance(adapter_key, str) or not adapter_key
        ):
            raise ScenarioError(
                f"programs[{index}].adapter_key must be a non-empty string"
            )
        if adapter_key is not None and adapter_key not in adapter_keys:
            raise ScenarioError(
                f"programs[{index}].adapter_key names unknown adapter {adapter_key!r}"
            )
        programs.append(
            Program(
                id=program_id,
                arrival_us=arrival_us,
                turns=tuple(turns),
                slo_us=slo_us,
                adapter_key=adapter_key,
            )
        )

    maximum_cache_bytes = max(
        sum(turn.new_input_tokens + turn.output_tokens for turn in program.turns)
        * config.kv_bytes_per_token
        for program in programs
    )
    if maximum_cache_bytes > max(instance.hbm_capacity_bytes for instance in instances):
        raise ScenarioError(
            "at least one program's KV cache is larger than every instance's HBM capacity"
        )
    adapter_sizes = {adapter.key: adapter.size_bytes for adapter in adapters}
    for adapter_key in sorted(
        {program.adapter_key for program in programs if program.adapter_key is not None}
    ):
        size_bytes = adapter_sizes[adapter_key]
        if size_bytes > max(instance.adapter_capacity_bytes for instance in instances):
            raise ScenarioError(
                f"adapter {adapter_key!r} is larger than every instance's adapter capacity"
            )
    return Workload(
        config=config,
        instances=tuple(instances),
        programs=tuple(programs),
        adapters=tuple(adapters),
    )


def load_workload(path: str | Path) -> Workload:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return workload_from_dict(raw)


def write_result(path: str | Path, result: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
