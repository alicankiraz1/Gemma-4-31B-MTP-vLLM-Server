from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app
from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.middleware import install_request_boundary_middleware
from gemma4_mtp_vllm.server.runtime_state import RuntimeState


def _make_app(limits: ServerLimits, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI()
    runtime_state = RuntimeState(max_queue_size=limits.max_queue_size)
    install_request_boundary_middleware(
        app,
        limits=limits,
        api_key=api_key,
        public_paths={"/livez"},
        runtime_state=runtime_state,
    )

    @app.get("/livez")
    async def livez() -> dict:
        return {"status": "ok"}

    @app.post("/protected")
    async def protected() -> dict:
        return {"ok": True}

    return app


def test_body_cap_enforced():
    limits = ServerLimits(max_body_bytes=16)
    client = TestClient(_make_app(limits))
    response = client.post("/protected", content="x" * 64)
    assert response.status_code == 413


def test_cors_default_deny_passes_no_origin_header():
    limits = ServerLimits()
    client = TestClient(_make_app(limits))
    response = client.options(
        "/protected",
        headers={"Origin": "https://example.com",
                 "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_opt_in_returns_origin_header():
    limits = ServerLimits(cors_origins=("https://app.test",))
    client = TestClient(_make_app(limits))
    response = client.options(
        "/protected",
        headers={"Origin": "https://app.test",
                 "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "https://app.test"


def test_cors_preflight_allowed_origin_bypasses_auth_and_rate_limit():
    limits = ServerLimits(cors_origins=("https://app.test",), rate_limit_rpm=1)
    client = TestClient(_make_app(limits, api_key="secret"))

    response = client.options(
        "/protected",
        headers={
            "Origin": "https://app.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )

    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == "https://app.test"
    assert response.headers["access-control-max-age"] == "600"
    assert "authorization" in response.headers["access-control-allow-headers"]
    assert "content-type" in response.headers["access-control-allow-headers"]
    assert "x-request-id" in response.headers
    assert client.post("/protected").status_code == 200


def test_allowed_origin_auth_error_includes_cors_headers():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    limits = ServerLimits(cors_origins=("https://app.test",))
    client = TestClient(
        create_app(
            api_key="secret",
            limits=limits,
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )

    response = client.get("/health", headers={"Origin": "https://app.test"})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "https://app.test"
    assert "x-request-id" in response.headers


def test_anthropic_body_cap_uses_anthropic_error_shape():
    limits = ServerLimits(max_body_bytes=16)
    client = TestClient(_make_app(limits))

    response = client.post("/v1/messages", content="x" * 64)

    assert response.status_code == 413
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "request_too_large"


def test_rate_limit_returns_429_after_threshold():
    limits = ServerLimits(rate_limit_rpm=2)
    client = TestClient(_make_app(limits))
    statuses = [client.post("/protected").status_code for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_public_paths_bypass_rate_limit_and_body_cap():
    limits = ServerLimits(rate_limit_rpm=1)
    client = TestClient(_make_app(limits))
    for _ in range(5):
        assert client.get("/livez").status_code == 200


def test_request_id_header_propagated():
    limits = ServerLimits()
    client = TestClient(_make_app(limits))
    response = client.post(
        "/protected",
        headers={"X-Request-ID": "abc-123"},
    )
    assert response.headers["x-request-id"] == "abc-123"
