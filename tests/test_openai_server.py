from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app
from gemma4_mtp_vllm.server.limits import ServerLimits


CAPTURED: dict = {}


def _vllm_handler(request: httpx.Request) -> httpx.Response:
    CAPTURED["path"] = request.url.path
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
            },
        )
    if request.url.path == "/version":
        return httpx.Response(200, json={"version": "0.11.0"})
    if request.url.path == "/v1/chat/completions":
        CAPTURED["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc",
                "object": "chat.completion",
                "model": "google/gemma-4-31B-it",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )
    return httpx.Response(404)


def _client():
    return TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(_vllm_handler),
        )
    )


def test_models_endpoint_returns_aliases():
    response = _client().get("/v1/models", headers={"x-api-key": "secret"})
    assert response.status_code == 200
    ids = {entry["id"] for entry in response.json()["data"]}
    assert "gemma-4-31b-mtp" in ids
    assert "claude-gemma-4-31b-mtp" in ids


def test_chat_completion_pass_through():
    CAPTURED.clear()
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hi"
    assert CAPTURED["body"]["model"] == "google/gemma-4-31B-it"
    assert CAPTURED["body"]["max_tokens"] == 32


def test_chat_completion_success_records_request_metrics():
    client = _client()
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )

    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text

    assert response.status_code == 200
    assert "gemma4_mtp_total_requests 1" in metrics
    assert "gemma4_mtp_active_requests 0" in metrics


def test_chat_completion_caps_max_tokens_at_limit():
    CAPTURED.clear()
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 99999,
        },
    )
    assert response.status_code == 200
    assert CAPTURED["body"]["max_tokens"] == 4096


def test_chat_completion_rejects_tools():
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"


@pytest.mark.parametrize("payload", [
    {"model": "gemma-4-31b-mtp", "messages": "abc"},
    {"model": [], "messages": [{"role": "user", "content": "hi"}]},
    {
        "model": "gemma-4-31b-mtp",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": "abc",
    },
])
def test_chat_completion_rejects_malformed_payload_without_forwarding(payload: dict):
    CAPTURED.clear()
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json=payload,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert CAPTURED == {}


def test_chat_completion_unknown_model_404():
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "not-a-thing",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404


def test_chat_completion_streaming_passthrough(monkeypatch):
    chunks_body = (
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunks_body,
            )
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"data: " in response.content
    assert b"[DONE]" in response.content


def test_chat_completion_streaming_releases_request_slot_after_consumption():
    chunks_body = (
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunks_body,
            )
        return httpx.Response(200, json={"status": "ok"})

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert b"[DONE]" in response.read()

    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text
    assert "gemma4_mtp_total_requests 1" in metrics
    assert "gemma4_mtp_active_requests 0" in metrics


def test_chat_completion_queue_full_rejects_without_forwarding():
    forwarded = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal forwarded
        if request.url.path == "/v1/chat/completions":
            forwarded = True
        return httpx.Response(200, json={"status": "ok"})

    app = create_app(
        api_key="secret",
        limits=ServerLimits(max_queue_size=1),
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )

    async def fill_queue():
        return [
            await app.state.runtime_state.acquire_generation_slot(),
            await app.state.runtime_state.acquire_generation_slot(),
        ]

    slots = asyncio.run(fill_queue())
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            headers={"x-api-key": "secret", "content-type": "application/json"},
            json={
                "model": "gemma-4-31b-mtp",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    finally:
        for slot in slots:
            slot.release()

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "queue_full"
    assert forwarded is False


def test_completions_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-abc",
                    "object": "text_completion",
                    "choices": [{"text": "World", "finish_reason": "stop", "index": 0}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(404)

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": "gemma-4-31b-mtp", "prompt": "Hello", "max_tokens": 4},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "World"


def test_completions_accepts_empty_string_prompt_and_forwards():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/completions":
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-empty",
                    "object": "text_completion",
                    "choices": [{"text": "World", "finish_reason": "stop", "index": 0}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(404)

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": "gemma-4-31b-mtp", "prompt": "", "max_tokens": 4},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "World"
    assert captured["body"]["prompt"] == ""


def test_completions_rejects_bad_max_tokens_without_forwarding():
    CAPTURED.clear()
    response = _client().post(
        "/v1/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "prompt": "Hello",
            "max_tokens": "abc",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert CAPTURED == {}
