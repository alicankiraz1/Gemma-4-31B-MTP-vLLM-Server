from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
