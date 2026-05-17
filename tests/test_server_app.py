from __future__ import annotations

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
    assert body["version_ok"] is False
    assert body["vllm"]["status"] == "ok"
    assert body["vllm"]["version"] == "0.11.0"


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
