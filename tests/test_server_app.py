from __future__ import annotations

import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client(api_key: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404)

    app = create_app(
        api_key=api_key,
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_livez_is_public():
    client = _client(api_key="secret")
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_requires_api_key():
    client = _client(api_key="secret")
    unauthorized = client.get("/readyz")
    authorized = client.get("/readyz", headers={"x-api-key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    body = authorized.json()
    assert body["status"] == "degraded"
    assert body["readiness"]["state"] == "degraded"
    assert "old_vllm_version" in body["readiness"]["reasons"]
    assert body["version_ok"] is False
    assert body["vllm"]["status"] == "ok"
    assert body["vllm"]["version"] == "0.11.0"


def test_readyz_degrades_when_target_model_is_not_served():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "wrong-model"}]})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    client = TestClient(app)

    health = client.get("/health", headers={"x-api-key": "secret"}).json()
    readyz = client.get("/readyz", headers={"x-api-key": "secret"}).json()

    assert health["status"] == "degraded"
    assert readyz["status"] == "degraded"
    assert health["readiness"]["state"] == readyz["readiness"]["state"]
    assert "target_not_served" in readyz["readiness"]["reasons"]
    assert readyz["target_served"] is False
    assert readyz["version_ok"] is True
    assert readyz["mtp"]["state"] == "unavailable"


def test_readyz_degrades_on_required_runtime_config_mismatch():
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
        "--quantization",
        "fp8",
        "--speculative-config",
        '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
    ]

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
                    'vllm:spec_decode_num_drafts_total{model_name="gemma-4-31b-mtp"} 2\n'
                    'vllm:spec_decode_num_draft_tokens_total{model_name="gemma-4-31b-mtp"} 8\n'
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="gemma-4-31b-mtp"} 6\n'
                ),
            )
        return httpx.Response(404)

    app = create_app(
        profile_name="tp2_2x32_fp8_gpuonly",
        api_key="secret",
        vllm_base_url="http://127.0.0.1:8000",
        vllm_transport=httpx.MockTransport(handler),
        runtime_manifest={"pid": 123, "argv": argv},
        active_backend_pid=123,
        active_backend_argv=argv,
    )
    body = TestClient(app).get("/readyz", headers={"x-api-key": "secret"}).json()

    assert body["status"] == "degraded"
    assert body["readiness"]["state"] == "degraded"
    assert "config_mismatch:cpu_offload_gb" in body["readiness"]["reasons"]
    assert body["config_verification"]["fields"]["cpu_offload_gb"]["status"] == "mismatch"


def test_readyz_reports_unavailable_when_backend_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/readyz", headers={"x-api-key": "secret"}).json()

    assert body["status"] == "unavailable"
    assert body["readiness"]["state"] == "unavailable"
    assert "backend_unreachable" in body["readiness"]["reasons"]


def test_readyz_reports_starting_when_backend_is_loading():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "loading"})
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    body = TestClient(app).get("/readyz", headers={"x-api-key": "secret"}).json()

    assert body["status"] == "starting"
    assert body["readiness"]["state"] == "starting"
    assert "backend_loading" in body["readiness"]["reasons"]


def test_readyz_returns_when_models_api_stalls(monkeypatch):
    monkeypatch.setattr(
        "gemma4_mtp_vllm.server.app.VLLM_READINESS_PROBE_TIMEOUT_SECONDS",
        0.01,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={"data": [{"id": "gemma-4-31b-mtp"}]})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )

    started = time.perf_counter()
    body = TestClient(app).get("/readyz", headers={"x-api-key": "secret"}).json()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.15
    assert body["status"] == "degraded"
    assert "target_not_served" in body["readiness"]["reasons"]


def test_readyz_returns_when_metrics_api_stalls(monkeypatch):
    monkeypatch.setattr(
        "gemma4_mtp_vllm.server.app.VLLM_READINESS_PROBE_TIMEOUT_SECONDS",
        0.01,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-4-31b-mtp"}]})
        if request.url.path == "/metrics":
            await asyncio.sleep(0.2)
            return httpx.Response(
                200,
                text="vllm:spec_decode_num_draft_tokens_total 0\n",
            )
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )

    started = time.perf_counter()
    body = TestClient(app).get("/readyz", headers={"x-api-key": "secret"}).json()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.15
    assert body["mtp"]["state"] == "unavailable"


def test_version_includes_gateway_and_vllm():
    client = _client(api_key="secret")
    response = client.get("/version", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["package"] == "gemma4-mtp-vllm"
    assert body["version"]
    assert body["vllm_version"] == "0.11.0"


def test_loopback_without_api_key_allowed():
    client = _client(api_key=None)
    assert client.get("/readyz").status_code == 200


def test_non_loopback_without_api_key_rejected():
    import pytest

    with pytest.raises(ValueError):
        create_app(
            api_key=None,
            bind_host="0.0.0.0",
            vllm_base_url="http://vllm.local:8000",
        )
