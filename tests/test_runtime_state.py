from __future__ import annotations

import asyncio

import pytest

from gemma4_mtp_vllm.server.runtime_state import (
    QueueFull,
    RuntimeState,
)


def test_initial_snapshot_is_zeroed():
    state = RuntimeState(max_queue_size=4)
    snapshot = state.snapshot()
    assert snapshot == {
        "active_requests": 0,
        "queued_requests": 0,
        "total_requests": 0,
        "rejected_requests": 0,
        "backend_errors": 0,
        "generation_tokens": 0,
        "generation_seconds": 0.0,
        "batch_requests": 0,
        "last_request_id": None,
    }


def test_record_generation_updates_counters():
    state = RuntimeState(max_queue_size=4)
    state.record_generation(
        generation_tokens=12,
        generation_seconds=0.5,
        batch_size=1,
    )
    snapshot = state.snapshot()
    assert snapshot["generation_tokens"] == 12
    assert snapshot["generation_seconds"] == pytest.approx(0.5)
    assert snapshot["batch_requests"] == 1


def test_record_backend_error_sets_last_error():
    state = RuntimeState(max_queue_size=2)
    state.record_backend_error("vllm_unreachable")
    snapshot = state.snapshot()
    assert state.last_backend_error == "vllm_unreachable"
    assert snapshot["backend_errors"] == 1
    state.clear_backend_error()
    assert state.last_backend_error is None


def test_acquire_generation_slot_bounded():
    async def scenario() -> None:
        state = RuntimeState(max_queue_size=1)
        slot1 = await state.acquire_generation_slot()
        slot2 = await state.acquire_generation_slot()
        with pytest.raises(QueueFull):
            await state.acquire_generation_slot()
        slot1.release()
        slot2.release()

    asyncio.run(scenario())


def test_request_id_recorded():
    state = RuntimeState(max_queue_size=2)
    state.note_request("req-1")
    assert state.snapshot()["last_request_id"] == "req-1"
