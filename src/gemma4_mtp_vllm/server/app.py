from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.anthropic_adapter import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
)
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.policy import (
    UnsupportedFeature,
    validate_anthropic_request,
    validate_openai_request,
)
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
from gemma4_mtp_vllm.server.runtime_state import QueueFull, RuntimeState
from gemma4_mtp_vllm.server.validation import (
    RequestValidationError,
    validate_anthropic_count_tokens_payload,
    validate_anthropic_messages_payload,
    validate_openai_chat_payload,
    validate_openai_completions_payload,
)

DEFAULT_MODEL_ALIAS = "gemma-4-31b-mtp"
DEFAULT_ANTHROPIC_MODEL_ALIAS = "claude-gemma-4-31b-mtp"
DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_PATHS = {"/livez"}


async def _bounded_json(
    request: Request,
    max_bytes: int,
    *,
    protocol: str = "openai",
) -> dict[str, Any] | JSONResponse:
    body = await request.body()
    if len(body) > max_bytes:
        return protocol_error_response(
            status_code=413,
            code="request_too_large",
            message=f"request body must be at most {max_bytes} bytes",
            protocol=protocol,
        )
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return protocol_error_response(
            status_code=400,
            code="invalid_request",
            message="request body must be valid JSON",
            protocol=protocol,
        )
    if not isinstance(parsed, dict):
        return protocol_error_response(
            status_code=400,
            code="invalid_request",
            message="request body must be a JSON object",
            protocol=protocol,
        )
    return parsed


def _alias_known(value: Any, aliases: Iterable[str]) -> bool:
    if not isinstance(value, str):
        return False
    return value in set(aliases)


def _validation_error_response(
    exc: RequestValidationError,
    *,
    protocol: str = "openai",
) -> JSONResponse:
    return protocol_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        protocol=protocol,
    )


async def _acquire_slot_or_error(
    runtime_state: RuntimeState,
    *,
    protocol: str = "openai",
) -> Any | JSONResponse:
    try:
        return await runtime_state.acquire_generation_slot()
    except QueueFull:
        return protocol_error_response(
            status_code=429,
            code="queue_full",
            message="generation queue is full",
            protocol=protocol,
        )


def _prepare_openai_body(
    payload: dict[str, Any],
    profile: ModelProfile,
    limits: ServerLimits,
) -> dict[str, Any]:
    body = dict(payload)
    body["model"] = profile.target
    requested_max = body.get("max_tokens", limits.max_output_tokens)
    if not isinstance(requested_max, int) or requested_max <= 0:
        requested_max = limits.max_output_tokens
    body["max_tokens"] = min(int(requested_max), limits.max_output_tokens)
    body.setdefault("temperature", profile.temperature)
    body.setdefault("top_p", profile.top_p)
    if profile.top_k > 0 and "top_k" not in body:
        body["top_k"] = profile.top_k
    body.pop("tools", None)
    body.pop("functions", None)
    body.pop("function_call", None)
    body.pop("tool_choice", None)
    body.pop("response_format", None)
    return body


async def openai_stream_to_anthropic_events_async(
    iterator,
    *,
    anthropic_model: str,
    message_id_prefix: str,
    prompt_tokens: int,
):
    # v0.1 trade-off: buffer the full OpenAI stream into memory before
    # converting to Anthropic events. The Anthropic translator is currently
    # synchronous; a true streaming async generator is a follow-up.
    chunks: list[dict] = []
    async for chunk in iterator:
        chunks.append(chunk)
    for event in openai_stream_to_anthropic_events(
        chunks,
        anthropic_model=anthropic_model,
        message_id_prefix=message_id_prefix,
        prompt_tokens=prompt_tokens,
    ):
        yield event


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

    @app.get("/health")
    async def health() -> dict[str, Any]:
        vllm_status = await _probe_vllm(vllm)
        if vllm_status.get("status") == "ok":
            try:
                version_body = await vllm.version()
                vllm_status["version"] = version_body.get("version")
            except VllmHttpError:
                vllm_status["version"] = None
        return {
            "status": "ready" if vllm_status.get("status") == "ok" else "degraded",
            "profile": selected.name,
            "target_model": selected.target,
            "drafter": selected.drafter,
            "num_speculative_tokens": selected.num_speculative_tokens,
            "tensor_parallel_size": selected.tensor_parallel_size,
            "gpu_memory_utilization": selected.gpu_memory_utilization,
            "max_model_len": selected.max_model_len,
            "model_aliases": aliases,
            "vllm": vllm_status,
            "bind": {"host": bind_host},
            "limits": server_limits.public_dict(),
            "runtime": runtime_state.snapshot(),
            "auth_modes": ["bearer", "x-api-key"],
            "tools_supported": False,
            "true_token_streaming": True,
            "continuous_batching": True,
            "token_counting": "estimated_word_count",
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        if not server_limits.metrics_enabled:
            return Response(status_code=404)
        snapshot = runtime_state.snapshot()
        body = (
            "# TYPE gemma4_mtp_active_requests gauge\n"
            f"gemma4_mtp_active_requests {snapshot['active_requests']}\n"
            "# TYPE gemma4_mtp_queued_requests gauge\n"
            f"gemma4_mtp_queued_requests {snapshot['queued_requests']}\n"
            "# TYPE gemma4_mtp_total_requests counter\n"
            f"gemma4_mtp_total_requests {snapshot['total_requests']}\n"
            "# TYPE gemma4_mtp_rejected_requests counter\n"
            f"gemma4_mtp_rejected_requests {snapshot['rejected_requests']}\n"
            "# TYPE gemma4_mtp_backend_errors counter\n"
            f"gemma4_mtp_backend_errors {snapshot['backend_errors']}\n"
            "# TYPE gemma4_mtp_generation_tokens_total counter\n"
            f"gemma4_mtp_generation_tokens_total {snapshot['generation_tokens']}\n"
            "# TYPE gemma4_mtp_generation_seconds_total counter\n"
            f"gemma4_mtp_generation_seconds_total {snapshot['generation_seconds']}\n"
            "# TYPE gemma4_mtp_batch_requests_total counter\n"
            f"gemma4_mtp_batch_requests_total {snapshot['batch_requests']}\n"
        )
        return Response(content=body, media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "owned_by": "local"}
                for alias in aliases
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await _bounded_json(request, server_limits.max_body_bytes)
        if isinstance(payload, JSONResponse):
            return payload
        try:
            validate_openai_chat_payload(payload)
            validate_openai_request(payload, mtp_enabled=True)
        except RequestValidationError as exc:
            return _validation_error_response(exc)
        except UnsupportedFeature as exc:
            return protocol_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
            )
        if not _alias_known(payload.get("model"), aliases):
            return protocol_error_response(
                status_code=404,
                code="model_not_found",
                message=f"model {payload.get('model')!r} is not available",
            )

        body = _prepare_openai_body(payload, selected, server_limits)
        streaming = bool(payload.get("stream"))
        slot = await _acquire_slot_or_error(runtime_state)
        if isinstance(slot, JSONResponse):
            return slot

        if streaming:
            async def event_stream():
                try:
                    async for chunk in vllm.chat_completion_stream(body):
                        if chunk.get("_done"):
                            yield "data: [DONE]\n\n"
                            return
                        yield f"data: {json.dumps(chunk)}\n\n"
                except VllmHttpError as exc:
                    runtime_state.record_backend_error("vllm_http_error")
                    error = {
                        "error": {
                            "code": "backend_unavailable",
                            "message": f"vllm returned {exc.status_code}",
                        }
                    }
                    yield f"data: {json.dumps(error)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    slot.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            response = await vllm.chat_completion(body)
        except VllmHttpError as exc:
            runtime_state.record_backend_error("vllm_http_error")
            return protocol_error_response(
                status_code=503,
                code="backend_unavailable",
                message=f"vllm returned {exc.status_code}",
            )
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=0.0,
                batch_size=1,
            )
            return JSONResponse(response)
        finally:
            slot.release()

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        payload = await _bounded_json(request, server_limits.max_body_bytes)
        if isinstance(payload, JSONResponse):
            return payload
        try:
            validate_openai_completions_payload(payload)
        except RequestValidationError as exc:
            return _validation_error_response(exc)
        if not _alias_known(payload.get("model"), aliases):
            return protocol_error_response(
                status_code=404,
                code="model_not_found",
                message=f"model {payload.get('model')!r} is not available",
            )
        body = dict(payload)
        body["model"] = selected.target
        body["max_tokens"] = min(
            int(body.get("max_tokens") or server_limits.max_output_tokens),
            server_limits.max_output_tokens,
        )
        slot = await _acquire_slot_or_error(runtime_state)
        if isinstance(slot, JSONResponse):
            return slot
        try:
            response = await vllm.completion(body)
        except VllmHttpError as exc:
            runtime_state.record_backend_error("vllm_http_error")
            return protocol_error_response(
                status_code=503,
                code="backend_unavailable",
                message=f"vllm returned {exc.status_code}",
            )
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=0.0,
                batch_size=1,
            )
            return JSONResponse(response)
        finally:
            slot.release()

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        payload = await _bounded_json(
            request,
            server_limits.max_body_bytes,
            protocol="anthropic",
        )
        if isinstance(payload, JSONResponse):
            return payload
        try:
            validate_anthropic_messages_payload(payload)
            validate_anthropic_request(payload)
        except RequestValidationError as exc:
            return _validation_error_response(exc, protocol="anthropic")
        except UnsupportedFeature as exc:
            return protocol_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                protocol="anthropic",
            )
        if not _alias_known(payload.get("model"), aliases):
            return protocol_error_response(
                status_code=404,
                code="model_not_found",
                message=f"model {payload.get('model')!r} is not available",
                protocol="anthropic",
            )

        openai_body = anthropic_request_to_openai(
            payload, openai_model=selected.target,
        )
        openai_body["max_tokens"] = min(
            int(openai_body.get("max_tokens") or server_limits.max_output_tokens),
            server_limits.max_output_tokens,
        )

        streaming = bool(payload.get("stream"))
        slot = await _acquire_slot_or_error(runtime_state, protocol="anthropic")
        if isinstance(slot, JSONResponse):
            return slot

        if streaming:
            async def event_stream():
                prompt_tokens = 0
                try:
                    async for event in openai_stream_to_anthropic_events_async(
                        vllm.chat_completion_stream(openai_body),
                        anthropic_model=payload.get("model")
                        or DEFAULT_ANTHROPIC_MODEL_ALIAS,
                        message_id_prefix="msg",
                        prompt_tokens=prompt_tokens,
                    ):
                        event_type = event.get("type", "message")
                        yield f"event: {event_type}\n"
                        yield f"data: {json.dumps(event)}\n\n"
                except VllmHttpError as exc:
                    runtime_state.record_backend_error("vllm_http_error")
                    err = {
                        "type": "error",
                        "error": {
                            "type": "backend_unavailable",
                            "message": f"vllm returned {exc.status_code}",
                        },
                    }
                    yield "event: error\n"
                    yield f"data: {json.dumps(err)}\n\n"
                finally:
                    slot.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            openai_response = await vllm.chat_completion(openai_body)
        except VllmHttpError as exc:
            runtime_state.record_backend_error("vllm_http_error")
            return protocol_error_response(
                status_code=503,
                code="backend_unavailable",
                message=f"vllm returned {exc.status_code}",
                protocol="anthropic",
            )
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (openai_response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=0.0,
                batch_size=1,
            )
            anthropic_body = openai_response_to_anthropic(
                openai_response,
                anthropic_model=payload.get("model") or DEFAULT_ANTHROPIC_MODEL_ALIAS,
                message_id_prefix="msg",
            )
            return JSONResponse(anthropic_body)
        finally:
            slot.release()

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> JSONResponse:
        payload = await _bounded_json(
            request,
            server_limits.max_body_bytes,
            protocol="anthropic",
        )
        if isinstance(payload, JSONResponse):
            return payload
        try:
            validate_anthropic_count_tokens_payload(payload)
            validate_anthropic_request(payload)
        except RequestValidationError as exc:
            return _validation_error_response(exc, protocol="anthropic")
        except UnsupportedFeature as exc:
            return protocol_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                protocol="anthropic",
            )
        text = " ".join(
            str(message.get("content", ""))
            for message in payload.get("messages", [])
            if isinstance(message, dict)
        )
        system = payload.get("system")
        if isinstance(system, str):
            text = f"{system} {text}"
        return JSONResponse(
            {"input_tokens": max(1, len(text.split()))},
            headers={"X-Gemma4-MTP-Token-Counting": "estimated_word_count"},
        )

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
