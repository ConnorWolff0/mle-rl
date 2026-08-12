from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AdapterReplicaView,
    CacheEntryView,
    ClusterState,
    InstanceConfig,
    InstanceState,
    PolicyViolation,
    Program,
    ProgramTrace,
    RequestTrace,
    RequestView,
    ScenarioError,
    SessionState,
    SimulationResult,
    ToolView,
    Turn,
    Workload,
)
from .policies import ServingPolicy


_COPY_COMPLETION = 0
_COMPLETION = 1
_ARRIVAL = 2


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _duration_us(units: int, units_per_second: int) -> int:
    if units == 0:
        return 0
    return _ceil_div(units * 1_000_000, units_per_second)


@dataclass(order=True)
class _Event:
    time_us: int
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class _Occupancy:
    instance_id: str
    tier: str
    size_bytes: int
    start_us: int
    end_us: int | None = None

    def active_at(self, time_us: int) -> bool:
        return self.start_us <= time_us and (
            self.end_us is None or time_us < self.end_us
        )


@dataclass
class _CacheEntry:
    session_id: str
    instance_id: str
    tier: str
    size_bytes: int
    tokens: int
    last_access_us: int
    completed_turns: int
    finished: bool
    tool: ToolView
    generation: int = 0
    available_us: int = 0
    occupancies: list[_Occupancy] = field(default_factory=list)


@dataclass
class _Session:
    program: Program
    arrived: bool = False
    completed_turns: int = 0
    cached_tokens: int = 0
    cache: _CacheEntry | None = None
    active_request_id: str | None = None
    request_ids: list[str] = field(default_factory=list)
    completion_us: int | None = None
    last_access_us: int = 0
    last_tool: ToolView = field(default_factory=lambda: ToolView("generic", None))


@dataclass
class _AdapterReplica:
    adapter_key: str
    instance_id: str
    size_bytes: int
    available_us: int
    last_access_us: int
    use_count: int
    occupancy: _Occupancy
    model_pinned_until_us: int = 0
    transfer_pinned_until_us: int = 0

    def pinned_until_us(self) -> int:
        return max(self.model_pinned_until_us, self.transfer_pinned_until_us)


@dataclass
class _QueuedRequest:
    view: RequestView
    turn: Turn
    adapter_replica: _AdapterReplica | None = None
    adapter_source: str = "none"
    adapter_ready_us: int = 0
    adapter_transfer_bytes: int = 0


@dataclass
class _RunningRequest:
    request: _QueuedRequest
    start_us: int
    transfer_us: int
    cache_source: str
    reused_tokens: int
    recomputed_tokens: int
    transfer_bytes: int
    adapter_source: str
    adapter_wait_us: int
    adapter_transfer_bytes: int
    service_us: int
    estimated_finish_us: int
    completion_pressure_us: int = 0
    output_finalized: bool = False


@dataclass
class _Instance:
    config: InstanceConfig
    queue: list[str] = field(default_factory=list)
    busy_until_us: int = 0
    running: _RunningRequest | None = None
    cache_copy_available_us: int = 0
    adapter_load_available_us: int = 0


class Simulator:
    def __init__(self, workload: Workload, policy: ServingPolicy):
        self.workload = workload
        self.policy = policy
        self.now_us = 0
        self._sequence = 0
        self._events: list[_Event] = []
        self._instances = {item.id: _Instance(item) for item in workload.instances}
        self._sessions = {item.id: _Session(item) for item in workload.programs}
        self._adapters = {item.key: item for item in workload.adapters}
        self._adapter_replicas: dict[tuple[str, str], _AdapterReplica] = {}
        self._retired_adapter_replicas: list[_AdapterReplica] = []
        self._retired_entries: list[_CacheEntry] = []
        self._requests: dict[str, _QueuedRequest] = {}
        self._request_traces: list[RequestTrace] = []
        self._completed_programs = 0
        self._metrics: dict[str, int] = {
            "cache_hbm_hits": 0,
            "cache_host_hits": 0,
            "cache_peer_hits": 0,
            "cache_misses": 0,
            "cold_starts": 0,
            "hbm_to_host_bytes": 0,
            "host_to_hbm_bytes": 0,
            "peer_transfer_bytes": 0,
            "peer_hbm_transfer_bytes": 0,
            "peer_host_transfer_bytes": 0,
            "hbm_pressure_demotions": 0,
            "hbm_pressure_evictions": 0,
            "host_pressure_evictions": 0,
            "finished_cache_evictions": 0,
            "recomputed_tokens": 0,
            "total_wait_us": 0,
            "total_service_us": 0,
            "total_transfer_us": 0,
            "adapter_requests": 0,
            "adapter_resident_hits": 0,
            "adapter_pending_hits": 0,
            "adapter_cold_loads": 0,
            "adapter_peer_copies": 0,
            "adapter_evictions": 0,
            "adapter_policy_evictions": 0,
            "adapter_cold_load_bytes": 0,
            "adapter_peer_transfer_bytes": 0,
            "total_adapter_transfer_us": 0,
            "total_adapter_wait_us": 0,
        }
        self._validate_workload()

    def _validate_workload(self) -> None:
        if not self.workload.instances:
            raise ScenarioError("workload must contain at least one instance")
        if not self.workload.programs:
            raise ScenarioError("workload must contain at least one program")
        instance_ids = [item.id for item in self.workload.instances]
        program_ids = [item.id for item in self.workload.programs]
        if len(instance_ids) != len(set(instance_ids)):
            raise ScenarioError("instance ids must be unique")
        if len(program_ids) != len(set(program_ids)):
            raise ScenarioError("program ids must be unique")
        adapter_keys = [item.key for item in self.workload.adapters]
        if len(adapter_keys) != len(set(adapter_keys)):
            raise ScenarioError("adapter keys must be unique")
        if any(not key or self._adapters[key].size_bytes <= 0 for key in adapter_keys):
            raise ScenarioError("adapter keys must be non-empty and sizes must be positive")
        config = self.workload.config
        numeric_config = (
            config.kv_bytes_per_token,
            config.slo_us,
            config.hbm_to_host_bytes_per_second,
            config.host_to_hbm_bytes_per_second,
            config.peer_bytes_per_second,
        )
        if (
            any(value <= 0 for value in numeric_config)
            or config.transfer_fixed_us < 0
            or config.adapter_peer_bytes_per_second <= 0
            or config.adapter_transfer_fixed_us < 0
        ):
            raise ScenarioError("rates, the default SLO, and KV bytes per token must be positive")
        for instance in self.workload.instances:
            if (
                instance.hbm_capacity_bytes <= 0
                or instance.host_capacity_bytes < 0
                or instance.prefill_tokens_per_second <= 0
                or instance.decode_tokens_per_second <= 0
                or instance.adapter_capacity_bytes < 0
                or instance.adapter_load_bytes_per_second <= 0
            ):
                raise ScenarioError(f"instance {instance.id!r} has an invalid hardware profile")
        for program in self.workload.programs:
            if program.arrival_us < 0 or not program.turns:
                raise ScenarioError(f"program {program.id!r} has an invalid arrival or no turns")
            if program.adapter_key is not None:
                if program.adapter_key not in self._adapters:
                    raise ScenarioError(
                        f"program {program.id!r} names unknown adapter {program.adapter_key!r}"
                    )
                adapter_size = self._adapters[program.adapter_key].size_bytes
                if adapter_size > max(
                    instance.adapter_capacity_bytes
                    for instance in self.workload.instances
                ):
                    raise ScenarioError(
                        f"adapter {program.adapter_key!r} does not fit on any instance"
                    )
            for turn in program.turns:
                if turn.new_input_tokens <= 0 or turn.output_tokens <= 0 or turn.tool_gap_us < 0:
                    raise ScenarioError(f"program {program.id!r} has an invalid turn")

    def run(self) -> SimulationResult:
        for program in sorted(self.workload.programs, key=lambda item: (item.arrival_us, item.id)):
            self._push(program.arrival_us, _ARRIVAL, "arrival", (program.id, 0))

        while self._events:
            event_time = self._events[0].time_us
            self.now_us = event_time
            while self._events and self._events[0].time_us == event_time:
                event = heapq.heappop(self._events)
                if event.kind == "arrival":
                    self._handle_arrival(*event.payload)
                elif event.kind == "completion":
                    self._handle_completion(event.payload)
                elif event.kind == "copy_completion":
                    self._handle_copy_completion()
                elif event.kind == "adapter_retry":
                    pass
                else:
                    raise AssertionError(f"unknown event {event.kind!r}")
            self._dispatch_idle_instances()

        expected_requests = sum(len(program.turns) for program in self.workload.programs)
        if self._completed_programs != len(self.workload.programs):
            raise RuntimeError("simulation ended with incomplete programs")
        if len(self._request_traces) != expected_requests:
            raise RuntimeError("simulation dropped or duplicated a request")
        return self._result()

    def _push(self, time_us: int, priority: int, kind: str, payload: Any) -> None:
        if time_us < self.now_us:
            raise RuntimeError("an event cannot be scheduled in the past")
        self._sequence += 1
        heapq.heappush(
            self._events,
            _Event(time_us, priority, self._sequence, kind, payload),
        )

    def _schedule_copy(
        self,
        source_instance_id: str,
        destination_instance_id: str,
        earliest_us: int,
        size_bytes: int,
        rate: int,
    ) -> tuple[int, int, int]:
        """Reserve both endpoint copy engines and return start, end, duration."""

        endpoint_ids = {source_instance_id, destination_instance_id}
        start_us = max(
            self.now_us,
            earliest_us,
            *(self._instances[item].cache_copy_available_us for item in endpoint_ids),
        )
        duration_us = self._transfer_duration(size_bytes, rate)
        end_us = start_us + duration_us
        for instance_id in endpoint_ids:
            self._instances[instance_id].cache_copy_available_us = end_us
        self._metrics["total_transfer_us"] += duration_us
        self._push(end_us, _COPY_COMPLETION, "copy_completion", None)
        return start_us, end_us, duration_us

    def _prepare_adapter(
        self,
        instance_id: str,
        request: _QueuedRequest,
        *,
        allow_defer: bool = False,
    ) -> _AdapterReplica | None:
        adapter_key = request.view.adapter_key
        if adapter_key is None:
            request.adapter_replica = None
            request.adapter_source = "none"
            request.adapter_ready_us = self.now_us
            request.adapter_transfer_bytes = 0
            return None

        replica = self._adapter_replicas.get((instance_id, adapter_key))
        if replica is not None:
            if request.adapter_replica is not replica:
                request.adapter_source = (
                    "resident" if replica.available_us <= self.now_us else "pending"
                )
                request.adapter_transfer_bytes = 0
            request.adapter_replica = replica
            request.adapter_ready_us = replica.available_us
            return replica

        size_bytes = self._adapters[adapter_key].size_bytes
        capacity_delay = self._ensure_adapter_capacity(
            instance_id,
            size_bytes,
            {adapter_key},
            allow_defer=allow_defer,
        )
        if capacity_delay is None:
            request.adapter_replica = None
            request.adapter_source = "deferred"
            request.adapter_ready_us = self.now_us
            request.adapter_transfer_bytes = 0
            return None
        capacity_ready_us = self.now_us + capacity_delay
        destination = self._instances[instance_id]
        fixed_us = self.workload.config.adapter_transfer_fixed_us
        cold_start_us = max(
            self.now_us,
            capacity_ready_us,
            destination.adapter_load_available_us,
        )
        cold_duration_us = fixed_us + _duration_us(
            size_bytes, destination.config.adapter_load_bytes_per_second
        )
        options: list[tuple[int, int, str, _AdapterReplica | None, int, int]] = [
            (
                cold_start_us + cold_duration_us,
                0,
                "cold",
                None,
                cold_start_us,
                cold_duration_us,
            )
        ]
        for source in sorted(
            self._adapter_replicas.values(),
            key=lambda item: (item.instance_id, item.adapter_key),
        ):
            if source.adapter_key != adapter_key or source.instance_id == instance_id:
                continue
            source_instance = self._instances[source.instance_id]
            peer_start_us = max(
                self.now_us,
                capacity_ready_us,
                source.available_us,
                source_instance.adapter_load_available_us,
                destination.adapter_load_available_us,
            )
            peer_duration_us = fixed_us + _duration_us(
                size_bytes, self.workload.config.adapter_peer_bytes_per_second
            )
            options.append(
                (
                    peer_start_us + peer_duration_us,
                    1,
                    source.instance_id,
                    source,
                    peer_start_us,
                    peer_duration_us,
                )
            )

        end_us, _kind_order, source_name, source, start_us, duration_us = min(
            options, key=lambda item: (item[0], item[1], item[2])
        )
        destination.adapter_load_available_us = end_us
        if source is None:
            adapter_source = "cold"
            self._metrics["adapter_cold_loads"] += 1
            self._metrics["adapter_cold_load_bytes"] += size_bytes
        else:
            adapter_source = f"peer:{source.instance_id}"
            source_instance = self._instances[source.instance_id]
            source_instance.adapter_load_available_us = end_us
            source.transfer_pinned_until_us = max(
                source.transfer_pinned_until_us, end_us
            )
            self._metrics["adapter_peer_copies"] += 1
            self._metrics["adapter_peer_transfer_bytes"] += size_bytes
        self._metrics["total_adapter_transfer_us"] += duration_us

        occupancy = _Occupancy(
            instance_id=instance_id,
            tier="adapter",
            size_bytes=size_bytes,
            start_us=start_us,
        )
        replica = _AdapterReplica(
            adapter_key=adapter_key,
            instance_id=instance_id,
            size_bytes=size_bytes,
            available_us=end_us,
            last_access_us=self.now_us,
            use_count=0,
            occupancy=occupancy,
        )
        self._adapter_replicas[(instance_id, adapter_key)] = replica
        request.adapter_replica = replica
        request.adapter_source = adapter_source
        request.adapter_ready_us = end_us
        request.adapter_transfer_bytes = size_bytes
        self._push(end_us, _COPY_COMPLETION, "copy_completion", None)
        self._assert_current_capacity()
        return replica

    def _ensure_adapter_capacity(
        self,
        instance_id: str,
        additional_bytes: int,
        protected: set[str],
        *,
        allow_defer: bool,
    ) -> int | None:
        capacity = self._instances[instance_id].config.adapter_capacity_bytes
        if additional_bytes > capacity:
            raise PolicyViolation(
                f"adapter needs {additional_bytes} bytes on {instance_id}, "
                f"whose adapter capacity is {capacity}"
            )
        naturally_ready = self._adapter_earliest_capacity_time(
            instance_id, additional_bytes
        )
        if naturally_ready is not None:
            return naturally_ready - self.now_us

        _ready_us, projected_peak = self._minimum_peak_adapter_usage_time(instance_id)
        deficit = projected_peak + additional_bytes - capacity
        candidates = tuple(
            self._adapter_view(replica)
            for replica in sorted(
                self._adapter_replicas.values(),
                key=lambda item: (item.adapter_key, item.instance_id),
            )
            if replica.instance_id == instance_id
            and replica.adapter_key not in protected
            and replica.available_us <= self.now_us
            and replica.transfer_pinned_until_us <= self.now_us
            and (
                not allow_defer
                or replica.model_pinned_until_us <= self.now_us
            )
        )
        if allow_defer and not candidates:
            retry_times = [
                max(replica.available_us, replica.pinned_until_us())
                for replica in self._adapter_replicas.values()
                if replica.instance_id == instance_id
                and replica.adapter_key not in protected
                and max(replica.available_us, replica.pinned_until_us()) > self.now_us
            ]
            if retry_times:
                self._push(min(retry_times), _COPY_COMPLETION, "adapter_retry", None)
            return None
        if not candidates and any(
            replica.instance_id == instance_id
            and replica.adapter_key not in protected
            and replica.available_us > self.now_us
            for replica in self._adapter_replicas.values()
        ):
            retry_us = min(
                max(replica.available_us, replica.pinned_until_us())
                for replica in self._adapter_replicas.values()
                if replica.instance_id == instance_id
                and replica.adapter_key not in protected
                and max(replica.available_us, replica.pinned_until_us()) > self.now_us
            )
            self._push(retry_us, _COPY_COMPLETION, "adapter_retry", None)
            return None
        order = self._adapter_eviction_order(
            instance_id, max(1, deficit), candidates
        )
        for adapter_key in order:
            naturally_ready = self._adapter_earliest_capacity_time(
                instance_id, additional_bytes
            )
            if naturally_ready is not None:
                return naturally_ready - self.now_us
            replica = self._adapter_replicas.get((instance_id, adapter_key))
            if replica is None:
                raise PolicyViolation(
                    f"adapter eviction candidate {adapter_key!r} is no longer resident"
                )
            self._remove_adapter_replica(replica)
            self._metrics["adapter_evictions"] += 1
        naturally_ready = self._adapter_earliest_capacity_time(
            instance_id, additional_bytes
        )
        if naturally_ready is None:
            retry_times = [
                max(
                    replica.available_us,
                    replica.transfer_pinned_until_us,
                    replica.model_pinned_until_us if allow_defer else 0,
                )
                for replica in self._adapter_replicas.values()
                if replica.instance_id == instance_id
                and replica.adapter_key not in protected
                and max(
                    replica.available_us,
                    replica.transfer_pinned_until_us,
                    replica.model_pinned_until_us if allow_defer else 0,
                )
                > self.now_us
            ]
            if retry_times:
                self._push(min(retry_times), _COPY_COMPLETION, "adapter_retry", None)
                return None
            raise PolicyViolation(
                f"evict did not free enough adapter memory on {instance_id}"
            )
        return naturally_ready - self.now_us

    def _adapter_eviction_order(
        self,
        instance_id: str,
        required_bytes: int,
        candidates: tuple[AdapterReplicaView, ...],
    ) -> list[str]:
        try:
            result = list(
                self.policy.evict_adapter(
                    instance_id,
                    required_bytes,
                    candidates,
                    self._state(),
                )
            )
        except Exception as error:
            raise PolicyViolation(
                f"evict_adapter failed on {instance_id}: {error}"
            ) from error
        valid_keys = {replica.adapter_key for replica in candidates}
        if (
            any(not isinstance(item, str) for item in result)
            or len(result) != len(set(result))
            or not set(result).issubset(valid_keys)
        ):
            raise PolicyViolation(
                "evict_adapter must return unique candidate adapter keys"
            )
        return result

    def _remove_adapter_replica(self, replica: _AdapterReplica) -> None:
        key = (replica.instance_id, replica.adapter_key)
        if self._adapter_replicas.get(key) is replica:
            del self._adapter_replicas[key]
        release_us = max(
            self.now_us, replica.available_us, replica.pinned_until_us()
        )
        replica.occupancy.end_us = release_us
        if release_us > self.now_us:
            self._retired_adapter_replicas.append(replica)

    def _adapter_physical_replicas(self) -> tuple[_AdapterReplica, ...]:
        return tuple(self._adapter_replicas.values()) + tuple(
            self._retired_adapter_replicas
        )

    def _adapter_used(self, instance_id: str, time_us: int | None = None) -> int:
        at_us = self.now_us if time_us is None else time_us
        return sum(
            replica.size_bytes
            for replica in self._adapter_physical_replicas()
            if replica.instance_id == instance_id
            and replica.occupancy.active_at(at_us)
        )

    def _adapter_usage_timeline(self, instance_id: str) -> list[tuple[int, int]]:
        used_bytes = self._adapter_used(instance_id, self.now_us)
        deltas: dict[int, int] = {}
        for replica in self._adapter_physical_replicas():
            occupancy = replica.occupancy
            if replica.instance_id != instance_id:
                continue
            if occupancy.start_us > self.now_us:
                deltas[occupancy.start_us] = (
                    deltas.get(occupancy.start_us, 0) + occupancy.size_bytes
                )
            if occupancy.end_us is not None and occupancy.end_us > self.now_us:
                deltas[occupancy.end_us] = (
                    deltas.get(occupancy.end_us, 0) - occupancy.size_bytes
                )
        timeline = [(self.now_us, used_bytes)]
        for time_us in sorted(deltas):
            used_bytes += deltas[time_us]
            timeline.append((time_us, used_bytes))
        return timeline

    def _adapter_earliest_capacity_time(
        self, instance_id: str, additional_bytes: int
    ) -> int | None:
        capacity = self._instances[instance_id].config.adapter_capacity_bytes
        suffix_peak = 0
        candidates: list[tuple[int, int]] = []
        for time_us, used_bytes in reversed(
            self._adapter_usage_timeline(instance_id)
        ):
            suffix_peak = max(suffix_peak, used_bytes)
            candidates.append((time_us, suffix_peak))
        for time_us, peak_bytes in reversed(candidates):
            if peak_bytes + additional_bytes <= capacity:
                return time_us
        return None

    def _minimum_peak_adapter_usage_time(self, instance_id: str) -> tuple[int, int]:
        suffix_peak = 0
        candidates: list[tuple[int, int]] = []
        for time_us, used_bytes in reversed(
            self._adapter_usage_timeline(instance_id)
        ):
            suffix_peak = max(suffix_peak, used_bytes)
            candidates.append((time_us, suffix_peak))
        return min(candidates, key=lambda item: (item[1], item[0]))

    def _move_cache(
        self,
        entry: _CacheEntry,
        destination_instance_id: str,
        destination_tier: str,
        *,
        reserve_us: int,
        earliest_us: int,
        rate: int,
        destination_size_bytes: int | None = None,
    ) -> tuple[int, int]:
        """Move one logical cache while retaining physical source occupancy."""

        source_instance_id = entry.instance_id
        source_tier = entry.tier
        open_sources = [
            occupancy
            for occupancy in entry.occupancies
            if occupancy.instance_id == source_instance_id
            and occupancy.tier == source_tier
            and occupancy.end_us is None
        ]
        if len(open_sources) != 1:
            raise RuntimeError("cache movement has no unique source occupancy")
        source = open_sources[0]
        earliest_us = max(earliest_us, entry.available_us, source.start_us)
        copy_bytes = entry.size_bytes
        destination_size = (
            copy_bytes
            if destination_size_bytes is None
            else destination_size_bytes
        )
        if not self._reservation_fits_from(
            destination_instance_id,
            destination_tier,
            destination_size,
            reserve_us,
        ):
            raise RuntimeError("cache destination was not reserved within capacity")
        _start_us, end_us, duration_us = self._schedule_copy(
            source_instance_id,
            destination_instance_id,
            earliest_us,
            copy_bytes,
            rate,
        )
        if not self.now_us <= reserve_us <= end_us:
            raise RuntimeError("cache destination reservation has invalid timing")
        source.end_us = end_us
        entry.occupancies.append(
            _Occupancy(
                instance_id=destination_instance_id,
                tier=destination_tier,
                size_bytes=destination_size,
                start_us=reserve_us,
            )
        )
        entry.instance_id = destination_instance_id
        entry.tier = destination_tier
        entry.size_bytes = destination_size
        entry.available_us = end_us
        entry.generation += 1
        self._assert_current_capacity()
        return end_us, duration_us

    def _resize_hbm_cache(
        self, entry: _CacheEntry, size_bytes: int, effective_us: int
    ) -> None:
        open_hbm = [
            occupancy
            for occupancy in entry.occupancies
            if occupancy.instance_id == entry.instance_id
            and occupancy.tier == "hbm"
            and occupancy.end_us is None
        ]
        if len(open_hbm) != 1:
            raise RuntimeError("HBM resize has no unique destination occupancy")
        current = open_hbm[0]
        if current.size_bytes == size_bytes:
            return
        additional_bytes = size_bytes - current.size_bytes
        if additional_bytes > 0 and not self._reservation_fits_from(
            entry.instance_id, "hbm", additional_bytes, effective_us
        ):
            raise RuntimeError("HBM resize was not reserved within capacity")
        current.end_us = effective_us
        entry.occupancies.append(
            _Occupancy(
                instance_id=entry.instance_id,
                tier="hbm",
                size_bytes=size_bytes,
                start_us=effective_us,
            )
        )
        entry.size_bytes = size_bytes
        self._assert_current_capacity()

    def _handle_copy_completion(self) -> None:
        for session in self._sessions.values():
            if session.cache is not None:
                session.cache.occupancies[:] = [
                    occupancy
                    for occupancy in session.cache.occupancies
                    if occupancy.end_us is None or occupancy.end_us > self.now_us
                ]
        self._retired_entries[:] = [
            entry
            for entry in self._retired_entries
            if any(
                occupancy.end_us is None or occupancy.end_us > self.now_us
                for occupancy in entry.occupancies
            )
        ]
        self._retired_adapter_replicas[:] = [
            replica
            for replica in self._retired_adapter_replicas
            if replica.occupancy.end_us is None
            or replica.occupancy.end_us > self.now_us
        ]

    def _handle_arrival(self, program_id: str, turn_index: int) -> None:
        session = self._sessions[program_id]
        if session.completed_turns != turn_index or session.active_request_id is not None:
            raise RuntimeError(f"request ordering violation for session {program_id!r}")
        if session.cache is not None:
            session.cache.generation += 1
        session.arrived = True

        turn = session.program.turns[turn_index]
        adapter_key = session.program.adapter_key
        adapter_size_bytes = (
            self._adapters[adapter_key].size_bytes if adapter_key is not None else 0
        )
        previous_tokens = session.cached_tokens
        hint = turn.output_token_hint if turn.output_token_hint is not None else turn.output_tokens
        request_id = f"{program_id}:{turn_index}"
        view = RequestView(
            id=request_id,
            program_id=program_id,
            turn_index=turn_index,
            ready_us=self.now_us,
            new_input_tokens=turn.new_input_tokens,
            output_token_hint=hint,
            previous_tokens=previous_tokens,
            cache_bytes_estimate=(previous_tokens + turn.new_input_tokens + hint)
            * self.workload.config.kv_bytes_per_token,
            deadline_us=session.program.arrival_us + self._program_slo(session.program),
            adapter_key=adapter_key,
            adapter_size_bytes=adapter_size_bytes,
        )
        request = _QueuedRequest(view=view, turn=turn)
        self._requests[request_id] = request
        session.active_request_id = request_id
        session.request_ids.append(request_id)

        try:
            route = self.policy.route(view, self._state())
        except Exception as error:
            raise PolicyViolation(f"route failed for {request_id}: {error}") from error
        if not isinstance(route, str) or route not in self._instances:
            raise PolicyViolation(f"route returned unknown instance {route!r} for {request_id}")
        self._instances[route].queue.append(request_id)
        self._prepare_adapter(route, request, allow_defer=True)

    def _dispatch_idle_instances(self) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            for instance_id in sorted(self._instances):
                instance = self._instances[instance_id]
                if instance.running is not None or instance.busy_until_us > self.now_us or not instance.queue:
                    continue
                queued = tuple(self._requests[request_id].view for request_id in instance.queue)
                try:
                    ordered = list(self.policy.schedule(instance_id, queued, self._state()))
                except Exception as error:
                    raise PolicyViolation(f"schedule failed on {instance_id}: {error}") from error
                expected = [request.id for request in queued]
                if (
                    any(not isinstance(item, str) for item in ordered)
                    or len(ordered) != len(expected)
                    or len(set(ordered)) != len(ordered)
                    or set(ordered) != set(expected)
                ):
                    raise PolicyViolation(
                        f"schedule on {instance_id} must return each queued request id exactly once"
                    )
                instance.queue[:] = ordered
                request_id = instance.queue.pop(0)
                if not self._start_request(instance, self._requests[request_id]):
                    instance.queue.insert(0, request_id)
                    continue
                made_progress = True

    def _start_request(self, instance: _Instance, request: _QueuedRequest) -> bool:
        session = self._sessions[request.view.program_id]
        if session.active_request_id != request.view.id:
            raise RuntimeError("a session request was started out of order")
        adapter_replica = self._prepare_adapter(instance.config.id, request)
        if request.view.adapter_key is not None and adapter_replica is None:
            return False
        adapter_wait_us = max(0, request.adapter_ready_us - self.now_us)
        previous_tokens = session.cached_tokens
        context_tokens = previous_tokens + request.turn.new_input_tokens
        context_bytes = context_tokens * self.workload.config.kv_bytes_per_token
        source = session.cache
        protected = {request.view.program_id}

        if source is None:
            cache_source = "cold" if previous_tokens == 0 else "miss"
            reused_tokens = 0
            recomputed_tokens = previous_tokens
            transfer_bytes = 0
            capacity_delay = self._ensure_hbm_capacity(
                instance.config.id, context_bytes, protected
            )
            ready_us = self.now_us + capacity_delay
            if not self._reservation_fits_from(
                instance.config.id, "hbm", context_bytes, ready_us
            ):
                raise RuntimeError("new HBM cache was not reserved within capacity")
            entry = _CacheEntry(
                session_id=session.program.id,
                instance_id=instance.config.id,
                tier="hbm",
                size_bytes=context_bytes,
                tokens=context_tokens,
                last_access_us=self.now_us,
                completed_turns=session.completed_turns,
                finished=False,
                tool=session.last_tool,
                generation=1,
                available_us=ready_us,
                occupancies=[
                    _Occupancy(
                        instance_id=instance.config.id,
                        tier="hbm",
                        size_bytes=context_bytes,
                        start_us=ready_us,
                    )
                ],
            )
            transfer_us = capacity_delay
        elif source.instance_id == instance.config.id and source.tier == "hbm":
            cache_source = "hbm"
            reused_tokens = source.tokens
            recomputed_tokens = 0
            transfer_bytes = 0
            additional = max(0, context_bytes - source.size_bytes)
            capacity_delay = self._ensure_hbm_capacity(
                instance.config.id, additional, protected
            )
            ready_us = self.now_us + capacity_delay
            entry = source
            self._resize_hbm_cache(entry, context_bytes, ready_us)
            entry.available_us = ready_us
            entry.generation += 1
            transfer_us = capacity_delay
        elif source.instance_id == instance.config.id and source.tier == "host":
            cache_source = "host"
            reused_tokens = source.tokens
            recomputed_tokens = 0
            transfer_bytes = source.size_bytes
            capacity_delay = self._ensure_hbm_capacity(
                instance.config.id, context_bytes, protected
            )
            capacity_ready_us = self.now_us + capacity_delay
            ready_us, _copy_us = self._move_cache(
                source,
                instance.config.id,
                "hbm",
                reserve_us=capacity_ready_us,
                earliest_us=capacity_ready_us,
                rate=self.workload.config.host_to_hbm_bytes_per_second,
                destination_size_bytes=context_bytes,
            )
            self._metrics["host_to_hbm_bytes"] += transfer_bytes
            entry = source
            transfer_us = ready_us - self.now_us
        else:
            source_tier = source.tier
            cache_source = f"peer_{source_tier}"
            reused_tokens = source.tokens
            recomputed_tokens = 0
            transfer_bytes = source.size_bytes
            capacity_delay = self._ensure_hbm_capacity(
                instance.config.id, context_bytes, protected
            )
            capacity_ready_us = self.now_us + capacity_delay
            transfer_rate = self.workload.config.peer_bytes_per_second
            if source_tier == "host":
                transfer_rate = min(
                    transfer_rate,
                    self.workload.config.host_to_hbm_bytes_per_second,
                )
            ready_us, _copy_us = self._move_cache(
                source,
                instance.config.id,
                "hbm",
                reserve_us=capacity_ready_us,
                earliest_us=capacity_ready_us,
                rate=transfer_rate,
                destination_size_bytes=context_bytes,
            )
            self._metrics["peer_transfer_bytes"] += transfer_bytes
            self._metrics[f"peer_{source_tier}_transfer_bytes"] += transfer_bytes
            entry = source
            transfer_us = ready_us - self.now_us

        if cache_source == "hbm":
            self._metrics["cache_hbm_hits"] += 1
        elif cache_source == "host":
            self._metrics["cache_host_hits"] += 1
        elif cache_source.startswith("peer_"):
            self._metrics["cache_peer_hits"] += 1
        elif cache_source == "cold":
            self._metrics["cold_starts"] += 1
        else:
            self._metrics["cache_misses"] += 1

        prefill_tokens = request.turn.new_input_tokens + recomputed_tokens
        compute_us = _duration_us(
            prefill_tokens, instance.config.prefill_tokens_per_second
        ) + _duration_us(request.turn.output_tokens, instance.config.decode_tokens_per_second)
        prerequisite_us = max(transfer_us, adapter_wait_us)
        service_us = prerequisite_us + compute_us
        estimated_compute_us = _duration_us(
            prefill_tokens, instance.config.prefill_tokens_per_second
        ) + _duration_us(
            request.view.output_token_hint, instance.config.decode_tokens_per_second
        )
        if service_us <= 0:
            raise RuntimeError("request service time must be positive")

        entry.tokens = context_tokens
        entry.last_access_us = self.now_us
        entry.completed_turns = session.completed_turns
        entry.finished = False
        entry.tool = session.last_tool
        session.cache = entry
        instance.running = _RunningRequest(
            request=request,
            start_us=self.now_us,
            transfer_us=transfer_us,
            cache_source=cache_source,
            reused_tokens=reused_tokens,
            recomputed_tokens=recomputed_tokens,
            transfer_bytes=transfer_bytes,
            adapter_source=request.adapter_source,
            adapter_wait_us=adapter_wait_us,
            adapter_transfer_bytes=request.adapter_transfer_bytes,
            service_us=service_us,
            estimated_finish_us=self.now_us + prerequisite_us + estimated_compute_us,
        )
        instance.busy_until_us = self.now_us + service_us
        if adapter_replica is not None:
            adapter_replica.last_access_us = self.now_us
            adapter_replica.use_count += 1
            adapter_replica.model_pinned_until_us = max(
                adapter_replica.model_pinned_until_us, instance.busy_until_us
            )
            self._metrics["adapter_requests"] += 1
            if request.adapter_source == "resident":
                self._metrics["adapter_resident_hits"] += 1
            elif request.adapter_source == "pending":
                self._metrics["adapter_pending_hits"] += 1
            self._metrics["total_adapter_wait_us"] += adapter_wait_us
        self._assert_current_capacity()
        self._push(instance.busy_until_us, _COMPLETION, "completion", instance.config.id)
        self._metrics["recomputed_tokens"] += recomputed_tokens
        return True

    def _handle_completion(self, instance_id: str) -> None:
        instance = self._instances[instance_id]
        running = instance.running
        if running is None or instance.busy_until_us != self.now_us:
            return
        request = running.request
        session = self._sessions[request.view.program_id]
        if session.cache is None or session.cache.instance_id != instance_id:
            raise RuntimeError("running request lost its cache reservation")

        entry = session.cache
        if not running.output_finalized:
            running.estimated_finish_us = self.now_us
            output_bytes = request.turn.output_tokens * self.workload.config.kv_bytes_per_token
            pressure_us = self._ensure_hbm_capacity(
                instance_id, output_bytes, {session.program.id}
            )
            self._resize_hbm_cache(
                entry, entry.size_bytes + output_bytes, self.now_us + pressure_us
            )
            entry.tokens += request.turn.output_tokens
            entry.available_us = self.now_us + pressure_us
            running.output_finalized = True
            running.completion_pressure_us = pressure_us
            if pressure_us:
                instance.busy_until_us = self.now_us + pressure_us
                running.estimated_finish_us = instance.busy_until_us
                self._push(
                    instance.busy_until_us,
                    _COMPLETION,
                    "completion",
                    instance.config.id,
                )
                return
        entry.last_access_us = self.now_us
        entry.completed_turns = session.completed_turns + 1
        entry.finished = entry.completed_turns == len(session.program.turns)
        entry.tool = ToolView(request.turn.tool_class, request.turn.tool_gap_hint_us)
        entry.generation += 1

        session.cached_tokens = entry.tokens
        session.completed_turns += 1
        session.last_access_us = self.now_us
        session.last_tool = entry.tool
        session.active_request_id = None
        finished = session.completed_turns == len(session.program.turns)

        total_service_us = running.service_us + running.completion_pressure_us
        finish_us = self.now_us
        trace = RequestTrace(
            request_id=request.view.id,
            program_id=request.view.program_id,
            turn_index=request.view.turn_index,
            instance_id=instance_id,
            ready_us=request.view.ready_us,
            start_us=running.start_us,
            finish_us=finish_us,
            wait_us=running.start_us - request.view.ready_us,
            service_us=total_service_us,
            transfer_us=running.transfer_us + running.completion_pressure_us,
            cache_source=running.cache_source,
            reused_tokens=running.reused_tokens,
            recomputed_tokens=running.recomputed_tokens,
            transfer_bytes=running.transfer_bytes,
            new_input_tokens=request.turn.new_input_tokens,
            output_tokens=request.turn.output_tokens,
            adapter_key=request.view.adapter_key,
            adapter_source=running.adapter_source,
            adapter_wait_us=running.adapter_wait_us,
            adapter_transfer_bytes=running.adapter_transfer_bytes,
        )
        self._request_traces.append(trace)
        self._metrics["total_wait_us"] += trace.wait_us
        self._metrics["total_service_us"] += trace.service_us

        instance.running = None
        instance.busy_until_us = finish_us
        if request.adapter_replica is not None:
            request.adapter_replica.model_pinned_until_us = max(
                request.adapter_replica.model_pinned_until_us, finish_us
            )

        if finished:
            session.completion_us = finish_us
            self._completed_programs += 1
        else:
            next_turn = session.completed_turns
            next_ready = finish_us + request.turn.tool_gap_us
            self._push(next_ready, _ARRIVAL, "arrival", (session.program.id, next_turn))

        self._apply_adapter_disposition(request)
        if finished:
            self._remove_cache(entry)
            self._metrics["finished_cache_evictions"] += 1

    def _apply_adapter_disposition(self, request: _QueuedRequest) -> None:
        replica = request.adapter_replica
        if replica is None:
            return
        if replica.model_pinned_until_us > self.now_us:
            raise RuntimeError("adapter disposition ran while its replica was in use")
        key = (replica.instance_id, replica.adapter_key)
        if self._adapter_replicas.get(key) is not replica:
            return
        view = self._adapter_view(replica)
        try:
            disposition = self.policy.adapter_disposition(view, self._state())
        except Exception as error:
            raise PolicyViolation(
                f"adapter_disposition failed for {replica.adapter_key!r} "
                f"on {replica.instance_id}: {error}"
            ) from error
        if disposition not in {"keep", "evict"}:
            raise PolicyViolation(
                "adapter_disposition must return keep or evict"
            )
        if disposition == "evict":
            self._remove_adapter_replica(replica)
            self._metrics["adapter_evictions"] += 1
            self._metrics["adapter_policy_evictions"] += 1

    def _ensure_hbm_capacity(
        self, instance_id: str, additional_bytes: int, protected: set[str]
    ) -> int:
        instance = self._instances[instance_id]
        if additional_bytes > instance.config.hbm_capacity_bytes:
            raise PolicyViolation(
                f"request needs {additional_bytes} additional HBM bytes on {instance_id}, "
                f"whose capacity is {instance.config.hbm_capacity_bytes}"
            )
        naturally_ready = self._earliest_capacity_time(
            instance_id, "hbm", additional_bytes
        )
        if naturally_ready is not None:
            return naturally_ready - self.now_us
        ready_us, projected_peak = self._minimum_peak_tier_usage_time(
            instance_id, "hbm"
        )
        deficit = projected_peak + additional_bytes - instance.config.hbm_capacity_bytes
        if deficit <= 0:
            return ready_us - self.now_us
        candidates = self._cache_candidates(instance_id, "hbm", protected)
        order = self._lru_cache_order(candidates)
        for session_id in order:
            if self._reservation_fits_from(
                instance_id, "hbm", additional_bytes, ready_us
            ):
                break
            entry = self._sessions[session_id].cache
            if entry is None or entry.instance_id != instance_id or entry.tier != "hbm":
                raise PolicyViolation(f"eviction candidate {session_id!r} is no longer in HBM")
            host_protected = protected | {session_id}
            if (
                entry.size_bytes <= instance.config.host_capacity_bytes
                and self._host_can_fit_after_evictions(
                    instance_id, entry.size_bytes, host_protected
                )
            ):
                host_delay = self._ensure_host_capacity(
                    instance_id, entry.size_bytes, host_protected
                )
                host_ready_us = self.now_us + host_delay
                copy_bytes = entry.size_bytes
                end_us, _copy_us = self._move_cache(
                    entry,
                    instance_id,
                    "host",
                    reserve_us=host_ready_us,
                    earliest_us=host_ready_us,
                    rate=self.workload.config.hbm_to_host_bytes_per_second,
                )
                ready_us = max(ready_us, end_us)
                self._metrics["hbm_to_host_bytes"] += copy_bytes
                self._metrics["hbm_pressure_demotions"] += 1
            else:
                self._remove_cache(entry)
                self._metrics["hbm_pressure_evictions"] += 1
        if not self._reservation_fits_from(
            instance_id, "hbm", additional_bytes, ready_us
        ):
            raise PolicyViolation(f"evict did not free enough HBM on {instance_id}")
        return ready_us - self.now_us

    def _host_can_fit_after_evictions(
        self, instance_id: str, additional_bytes: int, protected: set[str]
    ) -> bool:
        instance = self._instances[instance_id]
        if self._earliest_capacity_time(instance_id, "host", additional_bytes) is not None:
            return True
        reclaimable = sum(
            entry.size_bytes
            for entry in self._cache_candidate_entries(
                instance_id, "host", protected, include_pending=True
            )
        )
        _ready_us, minimum_peak = self._minimum_peak_tier_usage_time(
            instance_id, "host"
        )
        return (
            minimum_peak - reclaimable + additional_bytes
            <= instance.config.host_capacity_bytes
        )

    def _ensure_host_capacity(
        self, instance_id: str, additional_bytes: int, protected: set[str]
    ) -> int:
        instance = self._instances[instance_id]
        if additional_bytes > instance.config.host_capacity_bytes:
            raise PolicyViolation(
                f"cache needs {additional_bytes} host bytes on {instance_id}, "
                f"whose capacity is {instance.config.host_capacity_bytes}"
            )
        naturally_ready = self._earliest_capacity_time(
            instance_id, "host", additional_bytes
        )
        if naturally_ready is not None:
            return naturally_ready - self.now_us
        ready_us, projected_peak = self._minimum_peak_tier_usage_time(
            instance_id, "host"
        )
        deficit = projected_peak + additional_bytes - instance.config.host_capacity_bytes
        if deficit <= 0:
            return ready_us - self.now_us
        candidates = self._cache_candidates(
            instance_id, "host", protected, include_pending=True
        )
        order = self._lru_cache_order(candidates)
        for session_id in order:
            if self._reservation_fits_from(
                instance_id, "host", additional_bytes, ready_us
            ):
                break
            entry = self._sessions[session_id].cache
            if entry is None or entry.instance_id != instance_id or entry.tier != "host":
                raise PolicyViolation(f"eviction candidate {session_id!r} is no longer in host cache")
            self._remove_cache(entry)
            self._metrics["host_pressure_evictions"] += 1
            ready_us = max(ready_us, entry.available_us)
        if not self._reservation_fits_from(
            instance_id, "host", additional_bytes, ready_us
        ):
            raise PolicyViolation(f"evict did not free enough host cache on {instance_id}")
        return ready_us - self.now_us

    @staticmethod
    def _lru_cache_order(candidates: tuple[CacheEntryView, ...]) -> list[str]:
        return [
            entry.session_id
            for entry in sorted(
                candidates,
                key=lambda entry: (entry.last_access_us, entry.session_id),
            )
        ]

    def _cache_candidates(
        self,
        instance_id: str,
        tier: str,
        protected: set[str],
        *,
        include_pending: bool = False,
    ) -> tuple[CacheEntryView, ...]:
        return tuple(
            sorted(
                (
                    self._cache_view(entry)
                    for entry in self._cache_candidate_entries(
                        instance_id,
                        tier,
                        protected,
                        include_pending=include_pending,
                    )
                ),
                key=lambda item: item.session_id,
            )
        )

    def _cache_candidate_entries(
        self,
        instance_id: str,
        tier: str,
        protected: set[str],
        *,
        include_pending: bool = False,
    ) -> tuple[_CacheEntry, ...]:
        running_sessions = {
            instance.running.request.view.program_id
            for instance in self._instances.values()
            if instance.running is not None
        }
        return tuple(
            session.cache
            for session in self._sessions.values()
            if session.cache is not None
            and session.cache.instance_id == instance_id
            and session.cache.tier == tier
            and (include_pending or session.cache.available_us <= self.now_us)
            and session.program.id not in protected
            and session.program.id not in running_sessions
        )

    def _remove_cache(self, entry: _CacheEntry) -> None:
        session = self._sessions[entry.session_id]
        if session.cache is entry:
            session.cache = None
        release_us = max(self.now_us, entry.available_us)
        for occupancy in entry.occupancies:
            if occupancy.end_us is None:
                occupancy.end_us = release_us
        if any(
            occupancy.end_us is not None and occupancy.end_us > self.now_us
            for occupancy in entry.occupancies
        ):
            self._retired_entries.append(entry)
        entry.generation += 1

    def _physical_entries(self) -> tuple[_CacheEntry, ...]:
        active = tuple(
            session.cache
            for session in self._sessions.values()
            if session.cache is not None
        )
        return active + tuple(self._retired_entries)

    def _tier_used(self, instance_id: str, tier: str, time_us: int) -> int:
        return sum(
            occupancy.size_bytes
            for entry in self._physical_entries()
            for occupancy in entry.occupancies
            if occupancy.instance_id == instance_id
            and occupancy.tier == tier
            and occupancy.active_at(time_us)
        )

    def _hbm_used(self, instance_id: str, time_us: int | None = None) -> int:
        return self._tier_used(
            instance_id, "hbm", self.now_us if time_us is None else time_us
        )

    def _host_used(self, instance_id: str, time_us: int | None = None) -> int:
        return self._tier_used(
            instance_id, "host", self.now_us if time_us is None else time_us
        )

    def _earliest_capacity_time(
        self, instance_id: str, tier: str, additional_bytes: int
    ) -> int | None:
        capacity = (
            self._instances[instance_id].config.hbm_capacity_bytes
            if tier == "hbm"
            else self._instances[instance_id].config.host_capacity_bytes
        )
        timeline = self._tier_usage_timeline(instance_id, tier)
        suffix_peak = 0
        candidates: list[tuple[int, int]] = []
        for time_us, used_bytes in reversed(timeline):
            suffix_peak = max(suffix_peak, used_bytes)
            candidates.append((time_us, suffix_peak))
        for time_us, peak_bytes in reversed(candidates):
            if peak_bytes + additional_bytes <= capacity:
                return time_us
        return None

    def _minimum_peak_tier_usage_time(
        self, instance_id: str, tier: str
    ) -> tuple[int, int]:
        suffix_peak = 0
        candidates: list[tuple[int, int]] = []
        for time_us, used_bytes in reversed(
            self._tier_usage_timeline(instance_id, tier)
        ):
            suffix_peak = max(suffix_peak, used_bytes)
            candidates.append((time_us, suffix_peak))
        return min(candidates, key=lambda item: (item[1], item[0]))

    def _reservation_fits_from(
        self,
        instance_id: str,
        tier: str,
        additional_bytes: int,
        start_us: int,
    ) -> bool:
        capacity = (
            self._instances[instance_id].config.hbm_capacity_bytes
            if tier == "hbm"
            else self._instances[instance_id].config.host_capacity_bytes
        )
        peak_bytes = self._tier_used(instance_id, tier, start_us)
        peak_bytes = max(
            peak_bytes,
            *(
                used_bytes
                for time_us, used_bytes in self._tier_usage_timeline(
                    instance_id, tier
                )
                if time_us >= start_us
            ),
        )
        return peak_bytes + additional_bytes <= capacity

    def _tier_usage_timeline(
        self, instance_id: str, tier: str
    ) -> list[tuple[int, int]]:
        used_bytes = self._tier_used(instance_id, tier, self.now_us)
        deltas: dict[int, int] = {}
        for entry in self._physical_entries():
            for occupancy in entry.occupancies:
                if occupancy.instance_id != instance_id or occupancy.tier != tier:
                    continue
                if occupancy.start_us > self.now_us:
                    deltas[occupancy.start_us] = (
                        deltas.get(occupancy.start_us, 0) + occupancy.size_bytes
                    )
                if occupancy.end_us is not None and occupancy.end_us > self.now_us:
                    deltas[occupancy.end_us] = (
                        deltas.get(occupancy.end_us, 0) - occupancy.size_bytes
                    )
        timeline = [(self.now_us, used_bytes)]
        for time_us in sorted(deltas):
            used_bytes += deltas[time_us]
            timeline.append((time_us, used_bytes))
        return timeline

    def _assert_current_capacity(self) -> None:
        for instance in self._instances.values():
            for tier, capacity in (
                ("hbm", instance.config.hbm_capacity_bytes),
                ("host", instance.config.host_capacity_bytes),
            ):
                used = self._tier_used(instance.config.id, tier, self.now_us)
                if used > capacity:
                    raise RuntimeError(
                        f"{tier} occupancy exceeds capacity on {instance.config.id}: "
                        f"{used} > {capacity}"
                    )
            adapter_used = self._adapter_used(instance.config.id)
            if adapter_used > instance.config.adapter_capacity_bytes:
                raise RuntimeError(
                    f"adapter occupancy exceeds capacity on {instance.config.id}: "
                    f"{adapter_used} > {instance.config.adapter_capacity_bytes}"
                )
            for time_us, used_bytes in self._adapter_usage_timeline(
                instance.config.id
            ):
                if used_bytes < 0 or used_bytes > instance.config.adapter_capacity_bytes:
                    raise RuntimeError(
                        f"adapter reservation exceeds capacity on {instance.config.id} "
                        f"at {time_us}: {used_bytes} > "
                        f"{instance.config.adapter_capacity_bytes}"
                    )

    def _transfer_duration(self, size_bytes: int, rate: int) -> int:
        if size_bytes == 0:
            return 0
        return self.workload.config.transfer_fixed_us + _duration_us(size_bytes, rate)

    def _cache_view(self, entry: _CacheEntry) -> CacheEntryView:
        return CacheEntryView(
            session_id=entry.session_id,
            instance_id=entry.instance_id,
            tier=entry.tier,  # type: ignore[arg-type]
            size_bytes=entry.size_bytes,
            tokens=entry.tokens,
            last_access_us=entry.last_access_us,
            completed_turns=entry.completed_turns,
            finished=entry.finished,
            tool=entry.tool,
            available_us=entry.available_us,
        )

    def _adapter_view(self, replica: _AdapterReplica) -> AdapterReplicaView:
        return AdapterReplicaView(
            adapter_key=replica.adapter_key,
            instance_id=replica.instance_id,
            size_bytes=replica.size_bytes,
            available_us=replica.available_us,
            last_access_us=replica.last_access_us,
            use_count=replica.use_count,
            in_use=replica.pinned_until_us() > self.now_us,
        )

    def _state(self) -> ClusterState:
        instances = tuple(
            InstanceState(
                id=instance.config.id,
                estimated_available_us=(
                    max(self.now_us, instance.running.estimated_finish_us)
                    if instance.running is not None
                    else max(self.now_us, instance.busy_until_us)
                ),
                running_request_id=(
                    instance.running.request.view.id if instance.running is not None else None
                ),
                queued_request_ids=tuple(instance.queue),
                hbm_used_bytes=self._hbm_used(instance.config.id),
                hbm_capacity_bytes=instance.config.hbm_capacity_bytes,
                host_used_bytes=self._host_used(instance.config.id),
                host_capacity_bytes=instance.config.host_capacity_bytes,
                prefill_tokens_per_second=instance.config.prefill_tokens_per_second,
                decode_tokens_per_second=instance.config.decode_tokens_per_second,
                cache_copy_available_us=max(
                    self.now_us, instance.cache_copy_available_us
                ),
                adapter_used_bytes=self._adapter_used(instance.config.id),
                adapter_capacity_bytes=instance.config.adapter_capacity_bytes,
                adapter_load_bytes_per_second=(
                    instance.config.adapter_load_bytes_per_second
                ),
                adapter_load_available_us=max(
                    self.now_us, instance.adapter_load_available_us
                ),
            )
            for instance in sorted(self._instances.values(), key=lambda item: item.config.id)
        )
        sessions = tuple(
            SessionState(
                id=session.program.id,
                next_turn_index=session.completed_turns,
                completed_turns=session.completed_turns,
                cache_instance_id=(session.cache.instance_id if session.cache is not None else None),
                cache_tier=(session.cache.tier if session.cache is not None else None),  # type: ignore[arg-type]
                cache_bytes=(session.cache.size_bytes if session.cache is not None else 0),
                cached_tokens=session.cached_tokens,
                last_access_us=session.last_access_us,
                finished=session.completion_us is not None,
                tool=session.last_tool,
                deadline_us=session.program.arrival_us + self._program_slo(session.program),
                cache_available_us=(
                    session.cache.available_us if session.cache is not None else 0
                ),
            )
            for session in sorted(self._sessions.values(), key=lambda item: item.program.id)
            if session.arrived
        )
        cache_entries = tuple(
            self._cache_view(session.cache)
            for session in sorted(self._sessions.values(), key=lambda item: item.program.id)
            if session.cache is not None
        )
        adapter_replicas = tuple(
            self._adapter_view(replica)
            for replica in sorted(
                self._adapter_replicas.values(),
                key=lambda item: (item.adapter_key, item.instance_id),
            )
        )
        config = self.workload.config
        return ClusterState(
            now_us=self.now_us,
            instances=instances,
            sessions=sessions,
            cache_entries=cache_entries,
            programs_completed=self._completed_programs,
            programs_arrived=sum(session.arrived for session in self._sessions.values()),
            kv_bytes_per_token=config.kv_bytes_per_token,
            hbm_to_host_bytes_per_second=config.hbm_to_host_bytes_per_second,
            host_to_hbm_bytes_per_second=config.host_to_hbm_bytes_per_second,
            peer_bytes_per_second=config.peer_bytes_per_second,
            transfer_fixed_us=config.transfer_fixed_us,
            adapter_replicas=adapter_replicas,
            adapter_peer_bytes_per_second=config.adapter_peer_bytes_per_second,
            adapter_transfer_fixed_us=config.adapter_transfer_fixed_us,
        )

    def _program_slo(self, program: Program) -> int:
        return program.slo_us if program.slo_us is not None else self.workload.config.slo_us

    def _result(self) -> SimulationResult:
        program_traces: list[ProgramTrace] = []
        for session in sorted(self._sessions.values(), key=lambda item: item.program.id):
            if session.completion_us is None:
                raise RuntimeError(f"program {session.program.id} did not complete")
            latency = session.completion_us - session.program.arrival_us
            slo = self._program_slo(session.program)
            program_traces.append(
                ProgramTrace(
                    program_id=session.program.id,
                    arrival_us=session.program.arrival_us,
                    completion_us=session.completion_us,
                    latency_us=latency,
                    slo_us=slo,
                    within_slo=latency <= slo,
                    request_ids=tuple(session.request_ids),
                )
            )
        latencies = sorted(trace.latency_us for trace in program_traces)
        within_slo = sum(trace.within_slo for trace in program_traces)
        makespan_start = min(program.arrival_us for program in self.workload.programs)
        makespan_end = max(trace.completion_us for trace in program_traces)
        makespan = max(1, makespan_end - makespan_start)
        score = within_slo / len(program_traces)
        metrics: dict[str, int | float] = dict(self._metrics)
        metrics.update(
            {
                "programs_total": len(program_traces),
                "programs_completed": len(program_traces),
                "programs_within_slo": within_slo,
                "requests_completed": len(self._request_traces),
                "makespan_us": makespan,
                "mean_program_latency_us": sum(latencies) / len(latencies),
                "p50_program_latency_us": self._nearest_rank(latencies, 0.50),
                "p95_program_latency_us": self._nearest_rank(latencies, 0.95),
                "p99_program_latency_us": self._nearest_rank(latencies, 0.99),
                "slo_success_rate": score,
                "slo_success_numerator": within_slo,
                "slo_success_denominator": len(program_traces),
                "slo_goodput_per_minute": within_slo * 60_000_000 / makespan,
            }
        )
        return SimulationResult(
            score=score,
            metrics=metrics,
            request_traces=tuple(
                sorted(self._request_traces, key=lambda item: (item.finish_us, item.request_id))
            ),
            program_traces=tuple(program_traces),
        )

    @staticmethod
    def _nearest_rank(values: list[int], quantile: float) -> int:
        index = max(0, math.ceil(quantile * len(values)) - 1)
        return values[index]


def run_simulation(workload: Workload, policy: ServingPolicy) -> SimulationResult:
    return Simulator(workload, policy).run()
