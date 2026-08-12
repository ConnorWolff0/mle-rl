from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class ScenarioError(ValueError):
    """Raised when a scenario is malformed or cannot run on the cluster."""


class PolicyViolation(RuntimeError):
    """Raised when a policy returns an invalid or unsafe decision."""


@dataclass(frozen=True)
class ToolView:
    tool_class: str
    gap_hint_us: int | None


@dataclass(frozen=True)
class Turn:
    new_input_tokens: int
    output_tokens: int
    tool_gap_us: int = 0
    output_token_hint: int | None = None
    tool_class: str = "generic"
    tool_gap_hint_us: int | None = None


@dataclass(frozen=True)
class AdapterConfig:
    key: str
    size_bytes: int


@dataclass(frozen=True)
class Program:
    id: str
    arrival_us: int
    turns: tuple[Turn, ...]
    slo_us: int | None = None
    adapter_key: str | None = None


@dataclass(frozen=True)
class InstanceConfig:
    id: str
    hbm_capacity_bytes: int
    host_capacity_bytes: int
    prefill_tokens_per_second: int
    decode_tokens_per_second: int
    adapter_capacity_bytes: int = 0
    adapter_load_bytes_per_second: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    kv_bytes_per_token: int
    slo_us: int
    hbm_to_host_bytes_per_second: int
    host_to_hbm_bytes_per_second: int
    peer_bytes_per_second: int
    transfer_fixed_us: int = 0
    adapter_peer_bytes_per_second: int = 1
    adapter_transfer_fixed_us: int = 0


@dataclass(frozen=True)
class Workload:
    config: SimulationConfig
    instances: tuple[InstanceConfig, ...]
    programs: tuple[Program, ...]
    adapters: tuple[AdapterConfig, ...] = ()


@dataclass(frozen=True)
class RequestView:
    id: str
    program_id: str
    turn_index: int
    ready_us: int
    new_input_tokens: int
    output_token_hint: int
    previous_tokens: int
    cache_bytes_estimate: int
    deadline_us: int
    adapter_key: str | None = None
    adapter_size_bytes: int = 0


@dataclass(frozen=True)
class CacheEntryView:
    session_id: str
    instance_id: str
    tier: Literal["hbm", "host"]
    size_bytes: int
    tokens: int
    last_access_us: int
    completed_turns: int
    finished: bool
    tool: ToolView
    available_us: int = 0


@dataclass(frozen=True)
class AdapterReplicaView:
    adapter_key: str
    instance_id: str
    size_bytes: int
    available_us: int
    last_access_us: int
    use_count: int
    in_use: bool = False


@dataclass(frozen=True)
class InstanceState:
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
    cache_copy_available_us: int = 0
    adapter_used_bytes: int = 0
    adapter_capacity_bytes: int = 0
    adapter_load_bytes_per_second: int = 1
    adapter_load_available_us: int = 0


@dataclass(frozen=True)
class SessionState:
    id: str
    next_turn_index: int
    completed_turns: int
    cache_instance_id: str | None
    cache_tier: Literal["hbm", "host"] | None
    cache_bytes: int
    cached_tokens: int
    last_access_us: int
    finished: bool
    tool: ToolView
    deadline_us: int
    cache_available_us: int = 0


@dataclass(frozen=True)
class ClusterState:
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
    adapter_replicas: tuple[AdapterReplicaView, ...] = ()
    adapter_peer_bytes_per_second: int = 1
    adapter_transfer_fixed_us: int = 0

    def instance(self, instance_id: str) -> InstanceState:
        for instance in self.instances:
            if instance.id == instance_id:
                return instance
        raise KeyError(instance_id)

    def session(self, session_id: str) -> SessionState:
        for session in self.sessions:
            if session.id == session_id:
                return session
        raise KeyError(session_id)

    def adapter_replicas_on(self, instance_id: str) -> tuple[AdapterReplicaView, ...]:
        return tuple(
            replica
            for replica in self.adapter_replicas
            if replica.instance_id == instance_id
        )


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    program_id: str
    turn_index: int
    instance_id: str
    ready_us: int
    start_us: int
    finish_us: int
    wait_us: int
    service_us: int
    transfer_us: int
    cache_source: str
    reused_tokens: int
    recomputed_tokens: int
    transfer_bytes: int
    new_input_tokens: int
    output_tokens: int
    adapter_key: str | None = None
    adapter_source: str = "none"
    adapter_wait_us: int = 0
    adapter_transfer_bytes: int = 0


@dataclass(frozen=True)
class ProgramTrace:
    program_id: str
    arrival_us: int
    completion_us: int
    latency_us: int
    slo_us: int
    within_slo: bool
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    score: float
    metrics: dict[str, int | float]
    request_traces: tuple[RequestTrace, ...]
    program_traces: tuple[ProgramTrace, ...]

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "score": self.score,
            "metrics": dict(self.metrics),
            "programs": [asdict(trace) for trace in self.program_traces],
            "requests": [asdict(trace) for trace in self.request_traces],
        }
