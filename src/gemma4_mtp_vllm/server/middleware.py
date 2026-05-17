from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Iterable

from fastapi import FastAPI, Request
from fastapi.responses import Response

from gemma4_mtp_vllm.server.errors import protocol_error_response
from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.runtime_state import RuntimeState


_RATE_WINDOW_SECONDS = 60.0


def install_request_boundary_middleware(
    app: FastAPI,
    *,
    limits: ServerLimits,
    api_key: str | None,
    public_paths: Iterable[str],
    runtime_state: RuntimeState,
) -> None:
    public_set = set(public_paths)
    rate_buckets: dict[str, deque[float]] = defaultdict(deque)
    allowed_origins = set(limits.cors_origins)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        path = request.url.path
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        runtime_state.note_request(request_id)
        origin = request.headers.get("origin")

        if _is_preflight(request):
            return _preflight_response(origin, allowed_origins, request_id)

        if path not in public_set:
            protocol = _protocol_for_path(path)
            length_error = await _enforce_body_cap(
                request,
                max_bytes=limits.max_body_bytes,
                protocol=protocol,
            )
            if length_error is not None:
                return _stamp(
                    length_error,
                    request_id,
                    origin=origin,
                    allowed_origins=allowed_origins,
                )

            if limits.rate_limit_rpm > 0:
                key = _rate_key(request, api_key)
                if not _allow_rate(rate_buckets[key], limits.rate_limit_rpm):
                    return _stamp(
                        protocol_error_response(
                            status_code=429,
                            code="rate_limited",
                            message="too many requests",
                            protocol=protocol,
                        ),
                        request_id,
                        origin=origin,
                        allowed_origins=allowed_origins,
                    )

        response = await call_next(request)
        _apply_cors(response, origin, allowed_origins)
        response.headers["x-request-id"] = request_id
        return response


def _protocol_for_path(path: str) -> str:
    return "anthropic" if path.startswith("/v1/messages") else "openai"


def _is_preflight(request: Request) -> bool:
    return (
        request.method == "OPTIONS"
        and request.headers.get("origin") is not None
        and request.headers.get("access-control-request-method") is not None
    )


def _preflight_response(
    origin: str | None,
    allowed: set[str],
    request_id: str,
) -> Response:
    response = Response(status_code=204)
    _apply_cors(response, origin, allowed)
    response.headers["access-control-max-age"] = "600"
    response.headers["x-request-id"] = request_id
    return response


async def _enforce_body_cap(
    request: Request,
    *,
    max_bytes: int,
    protocol: str = "openai",
) -> Response | None:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                return protocol_error_response(
                    status_code=413,
                    code="request_too_large",
                    message=f"request body must be at most {max_bytes} bytes",
                    protocol=protocol,
                )
        except ValueError:
            return protocol_error_response(
                status_code=400,
                code="invalid_request",
                message="content-length header must be a number",
                protocol=protocol,
            )
    body = await request.body()
    if len(body) > max_bytes:
        return protocol_error_response(
            status_code=413,
            code="request_too_large",
            message=f"request body must be at most {max_bytes} bytes",
            protocol=protocol,
        )
    request._body = body  # noqa: SLF001 - re-cache for downstream handlers
    return None


def _allow_rate(bucket: deque[float], rpm_limit: int) -> bool:
    now = time.monotonic()
    while bucket and now - bucket[0] > _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= rpm_limit:
        return False
    bucket.append(now)
    return True


def _rate_key(request: Request, api_key: str | None) -> str:
    if api_key and request.headers.get("authorization") == f"Bearer {api_key}":
        return "credential:bearer"
    if api_key and request.headers.get("x-api-key") == api_key:
        return "credential:x-api-key"
    client = request.client.host if request.client else "unknown"
    return f"client:{client}"


def _apply_cors(response: Response, origin: str | None, allowed: set[str]) -> None:
    if origin and origin in allowed:
        response.headers["access-control-allow-origin"] = origin
        response.headers["access-control-allow-methods"] = (
            "GET, POST, OPTIONS"
        )
        response.headers["access-control-allow-headers"] = (
            "authorization, content-type, x-api-key, x-request-id"
        )


def _stamp(
    response: Response,
    request_id: str,
    *,
    origin: str | None = None,
    allowed_origins: set[str] | None = None,
) -> Response:
    if allowed_origins is not None:
        _apply_cors(response, origin, allowed_origins)
    response.headers["x-request-id"] = request_id
    return response
