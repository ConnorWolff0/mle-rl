from __future__ import annotations

from collections.abc import Sequence

from serving_sim.models import (
    AdapterReplicaView,
    ClusterState,
    RequestView,
)


class Policy:
    def route(self, request: RequestView, state: ClusterState) -> str:
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

    def schedule(
        self,
        instance_id: str,
        queued_requests: tuple[RequestView, ...],
        state: ClusterState,
    ) -> Sequence[str]:
        return [request.id for request in queued_requests]

    def adapter_disposition(
        self, replica: AdapterReplicaView, state: ClusterState
    ) -> str:
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
                key=lambda replica: (
                    replica.last_access_us,
                    replica.use_count,
                    replica.adapter_key,
                ),
            )
        ]
