from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


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
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"


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
