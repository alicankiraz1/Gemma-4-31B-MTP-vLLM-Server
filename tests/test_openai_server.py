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
    items = response.json()["data"]
    ids = {entry["id"] for entry in items}
    assert "gemma-4-31b-mtp" in ids
    assert "claude-gemma-4-31b-mtp" in ids
    for item in items:
        assert item["display_name"] == "Gemma 4 31B MTP vLLM"


def test_models_endpoint_redacts_private_alias():
    private_alias = "/" + "home" + "/homelander/private-alias"
    client = TestClient(
        create_app(
            api_key="secret",
            model_alias=private_alias,
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(_vllm_handler),
        )
    )

    response = client.get("/v1/models", headers={"x-api-key": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert "REDACTED_PATH" in {entry["id"] for entry in body["data"]}
    assert private_alias not in str(body)


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
    assert CAPTURED["body"]["model"] == "gemma-4-31b-mtp"
    assert CAPTURED["body"]["max_tokens"] == 32


def test_chat_completion_redacts_private_upstream_model_echo():
    private_alias = "/" + "home" + "/homelander/private-alias"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-private",
                    "object": "chat.completion",
                    "model": private_alias,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    client = TestClient(
        create_app(
            api_key="secret",
            model_alias=private_alias,
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": private_alias, "messages": [{"role": "user", "content": "hi"}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "REDACTED_PATH"
    assert private_alias not in str(body)


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
    generation_seconds = next(
        float(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith("gemma4_mtp_generation_seconds_total ")
    )
    assert generation_seconds > 0


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


def test_chat_completion_preserves_upstream_4xx_status():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too many tokens",
                    }
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text
    assert "gemma4_mtp_backend_errors 0" in metrics


def test_chat_completion_generation_timeout_returns_504():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={"choices": [], "usage": {}})
        return httpx.Response(200, json={"status": "ok"})

    client = TestClient(
        create_app(
            api_key="secret",
            limits=ServerLimits(generation_timeout_seconds=0.001),
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "backend_timeout"


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
    assert "gemma4_mtp_generation_tokens_total 1" in metrics
    generation_seconds = next(
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith("gemma4_mtp_generation_seconds_total ")
    )
    assert generation_seconds > 0


def test_chat_completion_streaming_releases_slot_when_mtp_probe_is_cancelled(monkeypatch):
    mtp_probe_calls = 0

    async def fake_mtp_state(*_args, **_kwargs):
        nonlocal mtp_probe_calls
        mtp_probe_calls += 1
        if mtp_probe_calls > 1:
            raise asyncio.CancelledError()
        return {"state": "active", "metrics_registered": True}

    monkeypatch.setattr(
        "gemma4_mtp_vllm.server.app._mtp_state_for_generation",
        fake_mtp_state,
    )
    chunks_body = (
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunks_body,
            )
        return httpx.Response(404)

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )

    try:
        client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "secret", "content-type": "application/json"},
            json={
                "model": "gemma-4-31b-mtp",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    except BaseException as exc:
        assert isinstance(exc, asyncio.CancelledError)

    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text
    assert "gemma4_mtp_active_requests 0" in metrics


def test_chat_completion_streaming_releases_slot_when_initial_mtp_probe_is_cancelled(monkeypatch):
    async def fake_mtp_state(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "gemma4_mtp_vllm.server.app._mtp_state_for_generation",
        fake_mtp_state,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"data: [DONE]\n\n",
            )
        return httpx.Response(404)

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )

    try:
        client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "secret", "content-type": "application/json"},
            json={
                "model": "gemma-4-31b-mtp",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    except RuntimeError as exc:
        assert str(exc) == "No response returned."
    except asyncio.CancelledError:
        pass

    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text
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


def test_completions_endpoint_redacts_private_upstream_model_echo():
    private_alias = "/" + "home" + "/homelander/private-alias"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-private",
                    "object": "text_completion",
                    "model": private_alias,
                    "choices": [{"text": "World", "finish_reason": "stop", "index": 0}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    client = TestClient(
        create_app(
            api_key="secret",
            model_alias=private_alias,
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": private_alias, "prompt": "Hello", "max_tokens": 4},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "REDACTED_PATH"
    assert private_alias not in str(body)


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
