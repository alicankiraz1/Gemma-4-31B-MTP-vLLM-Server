from __future__ import annotations

from dataclasses import dataclass


class QueueFull(Exception):
    pass


@dataclass
class _Slot:
    state: "RuntimeState"

    def release(self) -> None:
        self.state._release_slot()


class RuntimeState:
    def __init__(self, *, max_queue_size: int) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._max_queue_size = max_queue_size
        self._active = 0
        self._total = 0
        self._rejected = 0
        self._backend_errors = 0
        self._generation_tokens = 0
        self._generation_seconds = 0.0
        self._batch_requests = 0
        self._last_request_id: str | None = None
        self._last_backend_error: str | None = None

    @property
    def last_backend_error(self) -> str | None:
        return self._last_backend_error

    async def acquire_generation_slot(self) -> _Slot:
        # The gateway slot is a synchronous in-memory backstop: vLLM owns
        # real concurrency through continuous batching, so this layer only
        # bounds how many requests the gateway will admit before rejecting.
        # `max_queue_size + 1` gives one active plus `max_queue_size`
        # acceptable concurrent admissions before the next call raises.
        if self._active >= self._max_queue_size + 1:
            self._rejected += 1
            raise QueueFull()
        self._active += 1
        self._total += 1
        return _Slot(state=self)

    def _release_slot(self) -> None:
        if self._active > 0:
            self._active -= 1

    def record_generation(
        self,
        *,
        generation_tokens: int | None,
        generation_seconds: float,
        batch_size: int,
    ) -> None:
        if generation_tokens is not None and generation_tokens > 0:
            self._generation_tokens += int(generation_tokens)
        if generation_seconds > 0:
            self._generation_seconds += float(generation_seconds)
        if batch_size > 0:
            self._batch_requests += int(batch_size)

    def record_backend_error(self, code: str) -> None:
        self._backend_errors += 1
        self._last_backend_error = code

    def clear_backend_error(self) -> None:
        self._last_backend_error = None

    def note_request(self, request_id: str) -> None:
        self._last_request_id = request_id

    def snapshot(self) -> dict[str, object]:
        return {
            "active_requests": self._active,
            "queued_requests": 0,
            "total_requests": self._total,
            "rejected_requests": self._rejected,
            "backend_errors": self._backend_errors,
            "generation_tokens": self._generation_tokens,
            "generation_seconds": self._generation_seconds,
            "batch_requests": self._batch_requests,
            "last_request_id": self._last_request_id,
        }
