from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import (
    ModelProfile,
    ProfileSet,
    load_profiles,
    resolve_profile,
)
from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key
from gemma4_mtp_vllm.server.errors import protocol_error_response
from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.middleware import install_request_boundary_middleware
from gemma4_mtp_vllm.server.runtime_state import RuntimeState

DEFAULT_MODEL_ALIAS = "gemma-4-31b-mtp"
DEFAULT_ANTHROPIC_MODEL_ALIAS = "claude-gemma-4-31b-mtp"
DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_PATHS = {"/livez"}


def create_app(
    *,
    profile_name: str | None = None,
    profiles: ProfileSet | None = None,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    bind_host: str = DEFAULT_BIND_HOST,
    api_key: str | None = None,
    limits: ServerLimits | None = None,
    vllm_base_url: str = "http://127.0.0.1:8000",
    vllm_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    if bind_host_requires_api_key(bind_host) and not api_key:
        raise ValueError(f"bind_host {bind_host} requires api_key")

    server_limits = limits or ServerLimits()
    profile_set = load_profiles() if profiles is None else profiles
    selected = resolve_profile(profile_name, profile_set)
    runtime_state = RuntimeState(max_queue_size=server_limits.max_queue_size)
    aliases = _aliases(profile_set, selected, model_alias)

    if vllm_transport is not None:
        http = httpx.AsyncClient(transport=vllm_transport, base_url=vllm_base_url)
    else:
        http = httpx.AsyncClient(base_url=vllm_base_url, timeout=httpx.Timeout(120.0))
    vllm = VllmClient(http=http, base_url=vllm_base_url)

    app = FastAPI(title="Gemma 4 31B MTP vLLM Gateway")
    app.state.vllm = vllm
    app.state.profile = selected
    app.state.aliases = aliases
    app.state.runtime_state = runtime_state
    app.state.limits = server_limits
    app.state.api_key = api_key
    app.state.bind_host = bind_host

    install_request_boundary_middleware(
        app,
        limits=server_limits,
        api_key=api_key,
        public_paths=PUBLIC_PATHS,
        runtime_state=runtime_state,
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Callable[[Request], Any]):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        auth_error = _auth_error(request, api_key, request.url.path)
        if auth_error is not None:
            return auth_error
        return await call_next(request)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await vllm.aclose()

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        vllm_status = await _probe_vllm(vllm)
        readiness = "ready" if vllm_status.get("status") == "ok" else "degraded"
        return {
            "status": readiness,
            "vllm": vllm_status,
            "last_backend_error": runtime_state.last_backend_error,
        }

    @app.get("/version")
    async def version() -> dict[str, Any]:
        try:
            vllm_version = (await vllm.version()).get("version")
        except VllmHttpError:
            vllm_version = None
        return {
            "package": "gemma4-mtp-vllm",
            "version": __version__,
            "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
            "vllm_version": vllm_version,
        }

    return app


def _aliases(
    profiles: ProfileSet,
    profile: ModelProfile,
    canonical: str,
) -> list[str]:
    items = {
        alias
        for alias, name in profiles.aliases.items()
        if name == profile.name
    }
    items.add(canonical)
    items.add(DEFAULT_ANTHROPIC_MODEL_ALIAS)
    return sorted(items)


def _auth_error(request: Request, api_key: str | None, path: str) -> JSONResponse | None:
    if api_key is None:
        return None
    if request.headers.get("authorization") == f"Bearer {api_key}":
        return None
    if request.headers.get("x-api-key") == api_key:
        return None

    if path in {"/v1/messages", "/v1/messages/count_tokens"}:
        return protocol_error_response(
            status_code=401,
            code="unauthorized",
            message="missing or invalid API key",
            protocol="anthropic",
        )
    return protocol_error_response(
        status_code=401,
        code="unauthorized",
        message="missing or invalid API key",
    )


async def _probe_vllm(vllm: VllmClient) -> dict[str, Any]:
    try:
        body = await vllm.health()
        if isinstance(body, dict) and body.get("status") == "ok":
            return {"status": "ok"}
        return {"status": "degraded", "raw": body}
    except VllmHttpError as exc:
        return {"status": "unreachable", "http_status": exc.status_code}
    except httpx.HTTPError as exc:
        return {"status": "unreachable", "error": str(exc)}
