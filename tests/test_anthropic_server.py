from __future__ import annotations

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app
from gemma4_mtp_vllm.server.limits import ServerLimits


def _vllm(handler):
    return TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )


def test_anthropic_messages_returns_message_envelope():
    captured: dict = {}

    def handler(request):
        if request.url.path in {"/health"}:
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-x",
                    "choices": [
                        {
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

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "Hi"
    assert body["usage"]["input_tokens"] == 4
    assert body["usage"]["output_tokens"] == 1
    assert body["stop_reason"] == "end_turn"
    assert body["id"].startswith("msg_")
    forwarded = captured["body"]
    assert forwarded["model"] == "gemma-4-31b-mtp"
    assert forwarded["messages"][0] == {"role": "system", "content": "Be concise."}
    assert forwarded["max_tokens"] == 8


def test_anthropic_messages_success_records_request_metrics():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-x",
                    "choices": [
                        {
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

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 8,
        },
    )

    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text

    assert response.status_code == 200
    assert "gemma4_mtp_total_requests 1" in metrics
    assert "gemma4_mtp_active_requests 0" in metrics


def test_anthropic_messages_streaming():
    body = (
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        return httpx.Response(404)

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    body_bytes = response.content
    assert b"event: message_start" in body_bytes
    assert b"event: content_block_delta" in body_bytes
    assert b"event: message_stop" in body_bytes
    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text
    assert "gemma4_mtp_generation_tokens_total 1" in metrics
    generation_seconds = next(
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith("gemma4_mtp_generation_seconds_total ")
    )
    assert generation_seconds > 0


def test_anthropic_messages_queue_full_uses_anthropic_error_shape_without_forwarding():
    forwarded = False

    def handler(request):
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
            "/v1/messages",
            headers={"x-api-key": "secret", "content-type": "application/json"},
            json={
                "model": "claude-gemma-4-31b-mtp",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
            },
        )
    finally:
        for slot in slots:
            slot.release()

    body = response.json()
    assert response.status_code == 429
    assert body["type"] == "error"
    assert body["error"]["type"] == "queue_full"
    assert forwarded is False


def test_anthropic_rate_limit_uses_anthropic_error_shape():
    def handler(request):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-x",
                    "choices": [
                        {
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
        return httpx.Response(200, json={"status": "ok"})

    app = create_app(
        api_key="secret",
        limits=ServerLimits(rate_limit_rpm=1),
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    client = TestClient(app)
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
    }

    first = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json=payload,
    )
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json=payload,
    )

    assert first.status_code == 200
    assert response.status_code == 429
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "rate_limited"


def test_anthropic_count_tokens_uses_backend_tokenizer():
    captured: dict = {}

    def handler(request):
        if request.url.path == "/tokenize":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"count": 17, "tokens": list(range(17))})
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input_tokens"] == 17
    assert response.headers["x-gemma4-mtp-token-counting"] == "backend_tokenizer"
    assert captured["body"] == {
        "model": "gemma-4-31b-mtp",
        "messages": [{"role": "user", "content": "hello world"}],
    }


def test_anthropic_count_tokens_backend_error_is_503():
    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(503, json={"error": {"message": "down"}})
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "backend_unavailable"


def test_anthropic_count_tokens_rejects_unknown_model():
    forwarded = False

    def handler(request):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "not-a-model",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    body = response.json()

    assert response.status_code == 404
    assert body["type"] == "error"
    assert body["error"]["type"] == "model_not_found"
    assert forwarded is False


def test_anthropic_rejects_tools():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "calculator"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_feature"


def test_anthropic_messages_malformed_json_uses_anthropic_error_shape():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        content=b"{",
    )
    body = response.json()
    assert response.status_code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request"


def test_anthropic_messages_non_object_json_uses_anthropic_error_shape():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        content=b"[]",
    )
    body = response.json()
    assert response.status_code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request"


def test_anthropic_count_tokens_malformed_json_uses_anthropic_error_shape():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        content=b"{",
    )
    body = response.json()
    assert response.status_code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request"


def test_anthropic_count_tokens_non_object_json_uses_anthropic_error_shape():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        content=b"[]",
    )
    body = response.json()
    assert response.status_code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request"


def test_anthropic_messages_rejects_bad_max_tokens():
    forwarded = False

    def handler(request):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "max_tokens": "bad",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert forwarded is False


def test_anthropic_messages_rejects_non_list_messages():
    forwarded = False

    def handler(request):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "max_tokens": 4,
            "messages": "abc",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert forwarded is False


def test_anthropic_count_tokens_rejects_non_list_messages():
    forwarded = False

    def handler(request):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": "abc",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert forwarded is False
