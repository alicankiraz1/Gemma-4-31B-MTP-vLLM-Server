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
                text=(
                    "vllm:spec_decode_num_drafts_total 2\n"
                    "vllm:spec_decode_num_draft_tokens_total 8\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
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
    assert body["mtp"]["state"] == "active"
    assert body["mtp"]["drafted_tokens_total"] == 8.0
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

    argv = [
        "vllm",
        "serve",
        "google/gemma-4-31B-it",
        "--served-model-name",
        "gemma-4-31b-mtp",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.95",
        "--cpu-offload-gb",
        "8",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
        "--speculative-config",
        '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
    ]
    app = create_app(
        profile_name="tp2_2x32_smoke",
        api_key="secret",
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
        runtime_manifest={
            "pid": 123,
            "argv": argv,
            "package_versions": {"torch": "2.11.0+cu130"},
        },
        active_backend_pid=123,
        active_backend_argv=argv,
    )
    response = TestClient(app).get("/health", headers={"x-api-key": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["limits"]["max_output_tokens"] == 1024
    assert body["config_verification"]["status"] == "partial"
    assert body["config_verification"]["fields"]["cpu_offload_gb"]["status"] == "verified"
    assert body["config_matches"] is False


def test_health_reports_registered_but_idle_mtp_metrics():
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
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 0\n"
                    "vllm:spec_decode_num_draft_tokens_total 0\n"
                    "vllm:spec_decode_num_accepted_tokens_total 0\n"
                ),
            )
        return httpx.Response(404)

    app = create_app(
        profile_name="tp2_2x32_smoke",
        api_key="secret",
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/health", headers={"x-api-key": "secret"}).json()

    assert body["mtp"]["state"] == "registered_but_idle"
    assert body["mtp_observed"] is False


def test_health_reports_mtp_unavailable_when_metrics_endpoint_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-4-31b-mtp"}]})
        if request.url.path == "/metrics":
            return httpx.Response(404)
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/health", headers={"x-api-key": "secret"}).json()

    assert body["mtp"]["state"] == "unavailable"
    assert body["mtp_observed"] is False


def test_health_keeps_mtp_metrics_when_model_listing_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(503, json={"error": "loading"})
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    'vllm:spec_decode_num_drafts_total{model_name="gemma-4-31b-mtp"} 2\n'
                    'vllm:spec_decode_num_draft_tokens_total{model_name="gemma-4-31b-mtp"} 8\n'
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="gemma-4-31b-mtp"} 6\n'
                ),
            )
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/health", headers={"x-api-key": "secret"}).json()

    assert body["target_served"] is False
    assert body["mtp"]["state"] == "active"
    assert body["mtp"]["drafted_tokens_total"] == 8.0
    assert body["mtp_observed"] is True


def test_health_redacts_private_model_alias():
    private_alias = "/" + "home" + "/homelander/private-alias"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": private_alias}]})
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        model_alias=private_alias,
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/health", headers={"x-api-key": "secret"}).json()

    assert body["served_model_name"] == "REDACTED_PATH"
    assert "REDACTED_PATH" in body["model_aliases"]
    assert private_alias not in str(body)
