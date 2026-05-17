from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


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
    assert forwarded["model"] == "google/gemma-4-31B-it"
    assert forwarded["messages"][0] == {"role": "system", "content": "Be concise."}
    assert forwarded["max_tokens"] == 8


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


def test_anthropic_count_tokens_uses_word_count():
    def handler(request):
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
    assert body["input_tokens"] >= 2
    assert response.headers["x-gemma4-mtp-token-counting"] == "estimated_word_count"


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
