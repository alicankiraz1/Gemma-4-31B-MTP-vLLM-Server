from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client(api_key="secret"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [
                    {
                        "id": "gemma-4-31b-mtp",
                        "object": "model",
                        "max_model_len": 2048,
                    },
                    {"id": "google/gemma-4-31B-it-assistant", "object": "model"},
                ],
            })
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="vllm:spec_decode_draft_acceptance_rate 0.57\n",
            )
        return httpx.Response(404)

    app = create_app(
        api_key=api_key,
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_health_returns_profile_and_vllm_info():
    client = _client()
    response = client.get("/health", headers={"x-api-key": "secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["profile"] == "safe80"
    assert body["target_model"] == "google/gemma-4-31B-it"
    assert body["drafter"] == "google/gemma-4-31B-it-assistant"
    assert body["num_speculative_tokens"] == 4
    assert body["required_vllm_min_version"] == "0.21.0"
    assert body["version_ok"] is False
    assert body["vllm"]["status"] == "ok"
    assert body["vllm"]["version"] == "0.11.0"
    assert body["bind"]["host"] == "127.0.0.1"
    assert body["limits"]["max_output_tokens"] == 4096
    assert body["desired_config"]["max_model_len"] == 32768
    assert body["observed_config"]["max_model_len"] == 2048
    assert body["observed_config"]["target_served"] is True
    assert body["config_matches"] is False
    assert body["target_served"] is True
    assert body["mtp_observed"] is True
    assert body["runtime"]["total_requests"] == 0
    assert body["model_aliases"]
    assert body["tools_supported"] is False
    assert body["multimodal_supported"] is False
    assert body["streaming"] == {
        "openai": "vllm_passthrough_sse",
        "anthropic": "buffered_translation",
    }
    assert body["batching"] == {
        "backend": "vllm_continuous_batching",
        "gateway": "bounded_admission",
    }
    assert body["token_counting"] == "backend_tokenizer"
    assert "true_token_streaming" not in body
    assert "continuous_batching" not in body


def test_health_uses_profile_max_output_default_when_limits_not_overridden():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "gemma-4-31b-mtp", "max_model_len": 2048}]},
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(404)

    app = create_app(
        profile_name="tp2_2x32_smoke",
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    response = TestClient(app).get("/health", headers={"x-api-key": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["limits"]["max_output_tokens"] == 1024
    assert body["config_matches"] is True
