from __future__ import annotations

import json

import httpx
import pytest

from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError


def _client(handler) -> VllmClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://vllm.local:8000",
    )
    return VllmClient(http=http, base_url="http://vllm.local:8000")


@pytest.mark.asyncio
async def test_health_returns_status():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    async with _client(handler) as client:
        assert (await client.health())["status"] == "ok"


async def test_health_accepts_empty_success_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.local",
    ) as http:
        client = VllmClient(http=http, base_url="http://vllm.local")
        assert await client.health() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_models_returns_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
            },
        )

    async with _client(handler) as client:
        models = await client.list_models()
        assert models["data"][0]["id"] == "google/gemma-4-31B-it"


@pytest.mark.asyncio
async def test_metrics_text_returns_raw_plain_text():
    metrics = (
        "# TYPE vllm:spec_decode_draft_acceptance_rate gauge\n"
        "vllm:spec_decode_draft_acceptance_rate{model=\"gemma\"} 0.72\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(
            200,
            content=metrics,
            headers={"content-type": "text/plain; version=0.0.4"},
        )

    async with _client(handler) as client:
        assert await client.metrics_text() == metrics


@pytest.mark.asyncio
async def test_metrics_text_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(503, text="metrics unavailable")

    async with _client(handler) as client:
        with pytest.raises(VllmHttpError) as exc:
            await client.metrics_text()
        assert exc.value.status_code == 503
        assert str(exc.value) == "metrics unavailable"


@pytest.mark.asyncio
async def test_list_models_rejects_empty_success_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, content=b"")

    async with _client(handler) as client:
        with pytest.raises(json.JSONDecodeError):
            await client.list_models()


@pytest.mark.asyncio
async def test_chat_completion_proxies_body_and_returns_json():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "abc",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    async with _client(handler) as client:
        response = await client.chat_completion(
            {
                "model": "google/gemma-4-31B-it",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            }
        )
        assert response["choices"][0]["message"]["content"] == "hi"

    assert captured["body"]["max_tokens"] == 4


@pytest.mark.asyncio
async def test_chat_completion_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async with _client(handler) as client:
        with pytest.raises(VllmHttpError) as exc:
            await client.chat_completion({"messages": []})
        assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_chat_completion_stream_yields_chunks():
    body = (
        "data: {\"id\":\"a\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    async with _client(handler) as client:
        chunks = []
        async for chunk in client.chat_completion_stream({"messages": []}):
            chunks.append(chunk)
        assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
        assert chunks[-1] == {"_done": True}
