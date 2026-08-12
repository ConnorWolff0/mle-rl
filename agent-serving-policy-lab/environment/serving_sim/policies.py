from __future__ import annotations

from typing import Literal, Protocol, Sequence

from .models import (
    AdapterReplicaView,
    ClusterState,
    RequestView,
)


class ServingPolicy(Protocol):
    def route(self, request: RequestView, state: ClusterState) -> str:
        ...

    def schedule(
        self,
        instance_id: str,
        queued_requests: tuple[RequestView, ...],
        state: ClusterState,
    ) -> Sequence[str]:
        ...

    def adapter_disposition(
        self, replica: AdapterReplicaView, state: ClusterState
    ) -> Literal["keep", "evict"]:
        ...

    def evict_adapter(
        self,
        instance_id: str,
        required_bytes: int,
        candidates: tuple[AdapterReplicaView, ...],
        state: ClusterState,
    ) -> Sequence[str]:
        ...


class BaselinePolicy:
    """Sticky routing, FIFO scheduling, and adapter LRU under pressure."""

    def route(self, request: RequestView, state: ClusterState) -> str:
        session = state.session(request.program_id)
        if session.cache_instance_id is not None:
            return session.cache_instance_id
        return min(
            state.instances,
            key=lambda item: (
                len(item.queued_request_ids),
                item.estimated_available_us,
                item.id,
            ),
        ).id

    def schedule(
        self,
        instance_id: str,
        queued_requests: tuple[RequestView, ...],
        state: ClusterState,
    ) -> Sequence[str]:
        return [request.id for request in queued_requests]

    def adapter_disposition(
        self, replica: AdapterReplicaView, state: ClusterState
    ) -> Literal["keep", "evict"]:
        return "keep"

    def evict_adapter(
        self,
        instance_id: str,
        required_bytes: int,
        candidates: tuple[AdapterReplicaView, ...],
        state: ClusterState,
    ) -> Sequence[str]:
        return [
            replica.adapter_key
            for replica in sorted(
                candidates,
                key=lambda item: (
                    item.last_access_us,
                    item.use_count,
                    item.adapter_key,
                ),
            )
        ]
