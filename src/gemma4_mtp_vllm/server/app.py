from __future__ import annotations

import json
import asyncio
import time
from collections.abc import Callable, Iterable
from pathlib import Path
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
from gemma4_mtp_vllm.backend.response_parser import (
    ThoughtSanitizer,
    visible_text_for_history,
)
from gemma4_mtp_vllm.mtp_metrics import mtp_metric_delta
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
from gemma4_mtp_vllm.readiness import build_readiness_state
from gemma4_mtp_vllm.runtime_config import (
    build_config_verification,
    config_matches,
    default_observed_config,
    desired_config,
    merge_observed_config,
    observed_config_from_models,
    observed_config_from_metrics,
    observed_config_from_runtime_evidence,
    observed_config_from_version,
    public_observed_config,
    redact_public_value,
    read_text_tail,
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
from gemma4_mtp_vllm.versioning import version_at_least

DEFAULT_MODEL_ALIAS = "gemma-4-31b-mtp"
DEFAULT_ANTHROPIC_MODEL_ALIAS = "claude-gemma-4-31b-mtp"
MODEL_DISPLAY_NAME = "Gemma 4 31B MTP vLLM"
DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_PATHS = {"/livez"}
VLLM_READINESS_PROBE_TIMEOUT_SECONDS = 0.5
MTP_METRICS_PROBE_TIMEOUT_SECONDS = 0.5


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
    upstream_model: str,
) -> dict[str, Any]:
    body = dict(payload)
    body["messages"] = _sanitize_openai_messages(body.get("messages"))
    body["model"] = upstream_model
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


def _redact_openai_response(response: dict[str, Any]) -> dict[str, Any]:
    return redact_public_value(_sanitize_openai_response(response))


def _drop_openai_reasoning_fields(value: dict[str, Any]) -> None:
    value.pop("reasoning_content", None)
    value.pop("logprobs", None)


def _sanitize_openai_messages(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    messages: list[Any] = []
    for message in value:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        copied = dict(message)
        if copied.get("role") == "assistant":
            _drop_openai_reasoning_fields(copied)
            copied["content"] = _sanitize_openai_content(copied.get("content"))
        messages.append(copied)
    return messages


def _sanitize_openai_content(content: Any) -> Any:
    if isinstance(content, str):
        return visible_text_for_history(content)
    if isinstance(content, list):
        sanitizer = ThoughtSanitizer()
        sanitized_blocks: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                sanitized_blocks.append(block)
                continue
            copied = dict(block)
            _drop_openai_reasoning_fields(copied)
            if copied.get("type") == "text" and isinstance(copied.get("text"), str):
                copied["text"] = sanitizer.feed(copied["text"])
            sanitized_blocks.append(copied)
        tail = sanitizer.finish()
        if tail:
            for block in reversed(sanitized_blocks):
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = f"{block.get('text', '')}{tail}"
                    break
        return sanitized_blocks
    return content


def _sanitize_openai_response(response: dict[str, Any]) -> dict[str, Any]:
    result = dict(response)
    choices = response.get("choices")
    if not isinstance(choices, list):
        return result
    sanitized_choices: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            sanitized_choices.append(choice)
            continue
        copied_choice = dict(choice)
        _drop_openai_reasoning_fields(copied_choice)
        message = copied_choice.get("message")
        if isinstance(message, dict):
            copied_message = dict(message)
            _drop_openai_reasoning_fields(copied_message)
            if "content" in copied_message:
                copied_message["content"] = _sanitize_openai_content(
                    copied_message.get("content")
                )
            copied_choice["message"] = copied_message
        if isinstance(copied_choice.get("text"), str):
            copied_choice["text"] = visible_text_for_history(copied_choice["text"])
        sanitized_choices.append(copied_choice)
    result["choices"] = sanitized_choices
    return result


def _sanitize_openai_stream_chunk(
    chunk: dict[str, Any],
    sanitizers: dict[int, ThoughtSanitizer],
) -> dict[str, Any]:
    result = dict(chunk)
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return result
    sanitized_choices: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            sanitized_choices.append(choice)
            continue
        copied_choice = dict(choice)
        _drop_openai_reasoning_fields(copied_choice)
        index = copied_choice.get("index")
        sanitizer_index = index if isinstance(index, int) else 0
        sanitizer = sanitizers.setdefault(sanitizer_index, ThoughtSanitizer())
        delta = copied_choice.get("delta")
        copied_delta: dict[str, Any] | None = None
        if isinstance(delta, dict):
            copied_delta = dict(delta)
            _drop_openai_reasoning_fields(copied_delta)
            if isinstance(copied_delta.get("content"), str):
                content = sanitizer.feed(copied_delta["content"])
                if content:
                    copied_delta["content"] = content
                else:
                    copied_delta.pop("content", None)
        elif copied_choice.get("finish_reason") is not None:
            copied_delta = {}
        if copied_choice.get("finish_reason") is not None:
            final_content = sanitizer.finish()
            if final_content:
                if copied_delta is None:
                    copied_delta = {}
                existing_content = copied_delta.get("content")
                copied_delta["content"] = (
                    f"{existing_content}{final_content}"
                    if isinstance(existing_content, str)
                    else final_content
                )
        if copied_delta is not None:
            copied_choice["delta"] = copied_delta
        sanitized_choices.append(copied_choice)
    result["choices"] = sanitized_choices
    return result


def _openai_stream_content_chunk(index: int, content: str) -> dict[str, Any]:
    return {"choices": [{"index": index, "delta": {"content": content}}]}


async def openai_stream_to_anthropic_events_async(
    iterator,
    *,
    anthropic_model: str,
    message_id_prefix: str,
    prompt_tokens: int,
):
    # Current alpha trade-off: buffer the full OpenAI stream into memory before
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
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    active_backend_pid: int | None = None,
    active_backend_argv: list[str] | None = None,
    vllm_log_path: Path | None = None,
) -> FastAPI:
    if bind_host_requires_api_key(bind_host) and not api_key:
        raise ValueError(f"bind_host {bind_host} requires api_key")

    profile_set = load_profiles() if profiles is None else profiles
    selected = resolve_profile(profile_name, profile_set)
    server_limits = limits or ServerLimits(max_output_tokens=selected.max_output_tokens)
    runtime_state = RuntimeState(max_queue_size=server_limits.max_queue_size)
    aliases = _aliases(profile_set, selected, model_alias)
    served_model_name = model_alias

    if vllm_transport is not None:
        http = httpx.AsyncClient(transport=vllm_transport, base_url=vllm_base_url)
    else:
        http = httpx.AsyncClient(
            base_url=vllm_base_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        )
    vllm = VllmClient(http=http, base_url=vllm_base_url)

    app = FastAPI(title="Gemma 4 31B MTP vLLM Gateway")
    app.state.vllm = vllm
    app.state.profile = selected
    app.state.aliases = aliases
    app.state.runtime_state = runtime_state
    app.state.limits = server_limits
    app.state.api_key = api_key
    app.state.bind_host = bind_host

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Callable[[Request], Any]):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        auth_error = _auth_error(request, api_key, request.url.path)
        if auth_error is not None:
            return auth_error
        return await call_next(request)

    install_request_boundary_middleware(
        app,
        limits=server_limits,
        api_key=api_key,
        public_paths=PUBLIC_PATHS,
        runtime_state=runtime_state,
    )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await vllm.aclose()

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    async def _current_readiness_context() -> dict[str, Any]:
        vllm_status, version_ok = await _probe_vllm_with_version(vllm)
        desired = desired_config(selected, served_model_name=served_model_name)
        observed = await _observed_backend_config(
            vllm,
            selected,
            served_model_name=served_model_name,
            vllm_version=vllm_status.get("version"),
            runtime_manifest_path=runtime_manifest_path,
            runtime_manifest=runtime_manifest,
            active_backend_pid=active_backend_pid,
            active_backend_argv=active_backend_argv,
            vllm_log_path=vllm_log_path,
        )
        matches = config_matches(desired, observed)
        verification = build_config_verification(desired, observed)
        public_observed = public_observed_config(observed)
        runtime = runtime_state.snapshot()
        readiness = build_readiness_state(
            profile=selected,
            vllm_status=vllm_status,
            version_ok=version_ok,
            target_served=bool(public_observed.get("target_served", False)),
            config_verification=verification,
            mtp=public_observed.get("mtp"),
            runtime=runtime,
            last_backend_error=runtime_state.last_backend_error,
        )
        return {
            "vllm_status": vllm_status,
            "version_ok": version_ok,
            "desired": desired,
            "observed": public_observed,
            "config_verification": verification,
            "config_matches": matches,
            "runtime": runtime,
            "readiness": readiness,
        }

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        context = await _current_readiness_context()
        observed = context["observed"]
        return {
            "status": context["readiness"]["state"],
            "readiness": context["readiness"],
            "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
            "version_ok": context["version_ok"],
            "vllm": context["vllm_status"],
            "target_served": observed.get("target_served", False),
            "config_verification": context["config_verification"],
            "mtp": observed.get("mtp"),
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
        context = await _current_readiness_context()
        public_observed = context["observed"]
        return {
            "status": context["readiness"]["state"],
            "readiness": context["readiness"],
            "profile": selected.name,
            "target_model": redact_public_value(selected.target),
            "served_model_name": redact_public_value(served_model_name),
            "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
            "version_ok": context["version_ok"],
            "drafter": redact_public_value(selected.drafter),
            "num_speculative_tokens": selected.num_speculative_tokens,
            "tensor_parallel_size": selected.tensor_parallel_size,
            "gpu_memory_utilization": selected.gpu_memory_utilization,
            "max_model_len": selected.max_model_len,
            "desired_config": context["desired"],
            "observed_config": public_observed,
            "config_verification": context["config_verification"],
            "config_matches": context["config_matches"],
            "target_served": public_observed.get("target_served", False),
            "mtp": public_observed.get("mtp"),
            "mtp_observed": public_observed.get("mtp_observed", False),
            "model_aliases": redact_public_value(aliases),
            "vllm": context["vllm_status"],
            "bind": {"host": bind_host},
            "limits": server_limits.public_dict(),
            "runtime": context["runtime"],
            "auth_modes": ["bearer", "x-api-key"],
            "tools_supported": False,
            "multimodal_supported": False,
            "streaming": {
                "openai": "vllm_passthrough_sse",
                "anthropic": "buffered_translation",
            },
            "batching": {
                "backend": "vllm_continuous_batching",
                "gateway": "bounded_admission",
            },
            "token_counting": "backend_tokenizer",
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
                {
                    "id": redact_public_value(alias),
                    "object": "model",
                    "owned_by": "local",
                    "display_name": MODEL_DISPLAY_NAME,
                }
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

        body = _prepare_openai_body(
            payload,
            selected,
            server_limits,
            upstream_model=served_model_name,
        )
        streaming = bool(payload.get("stream"))
        slot = await _acquire_slot_or_error(runtime_state)
        if isinstance(slot, JSONResponse):
            return slot

        if streaming:
            try:
                mtp_before = await _mtp_state_for_generation(vllm, served_model_name)
            except BaseException:
                slot.release()
                raise
            stream_sanitizers: dict[int, ThoughtSanitizer] = {}

            async def event_stream():
                start = time.perf_counter()
                delta_tokens = 0
                usage_tokens: int | None = None
                stream_completed = False
                try:
                    async for chunk in vllm.chat_completion_stream(body):
                        if chunk.get("_done"):
                            for index, sanitizer in stream_sanitizers.items():
                                final_content = sanitizer.finish()
                                if final_content:
                                    final_chunk = _openai_stream_content_chunk(
                                        index,
                                        final_content,
                                    )
                                    delta_tokens += 1
                                    yield (
                                        f"data: {json.dumps(_redact_openai_response(final_chunk))}\n\n"
                                    )
                            stream_completed = True
                            yield "data: [DONE]\n\n"
                            return
                        chunk = _sanitize_openai_stream_chunk(chunk, stream_sanitizers)
                        usage_tokens = _stream_usage_tokens(chunk, usage_tokens)
                        if _chunk_has_content_delta(chunk):
                            delta_tokens += 1
                        yield f"data: {json.dumps(_redact_openai_response(chunk))}\n\n"
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
                    try:
                        if stream_completed or delta_tokens > 0 or usage_tokens is not None:
                            runtime_state.record_generation(
                                generation_tokens=(
                                    usage_tokens
                                    if usage_tokens is not None
                                    else delta_tokens
                                ),
                                generation_seconds=time.perf_counter() - start,
                                batch_size=1,
                            )
                            mtp_after = await _mtp_state_for_generation(
                                vllm,
                                served_model_name,
                            )
                            runtime_state.record_mtp_delta(
                                mtp_metric_delta(mtp_before, mtp_after)
                            )
                    finally:
                        slot.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            mtp_before = await _mtp_state_for_generation(vllm, served_model_name)
            start = time.perf_counter()
            response = await _with_generation_timeout(
                vllm.chat_completion(body),
                server_limits.generation_timeout_seconds,
            )
            elapsed = time.perf_counter() - start
        except asyncio.TimeoutError:
            runtime_state.record_backend_error("generation_timeout")
            return protocol_error_response(
                status_code=504,
                code="backend_timeout",
                message="generation exceeded gateway timeout",
            )
        except VllmHttpError as exc:
            return _vllm_error_response(exc, runtime_state=runtime_state)
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=elapsed,
                batch_size=1,
            )
            mtp_after = await _mtp_state_for_generation(vllm, served_model_name)
            runtime_state.record_mtp_delta(mtp_metric_delta(mtp_before, mtp_after))
            return JSONResponse(_redact_openai_response(response))
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
        body["model"] = served_model_name
        body["max_tokens"] = min(
            int(body.get("max_tokens") or server_limits.max_output_tokens),
            server_limits.max_output_tokens,
        )
        slot = await _acquire_slot_or_error(runtime_state)
        if isinstance(slot, JSONResponse):
            return slot
        try:
            mtp_before = await _mtp_state_for_generation(vllm, served_model_name)
            start = time.perf_counter()
            response = await _with_generation_timeout(
                vllm.completion(body),
                server_limits.generation_timeout_seconds,
            )
            elapsed = time.perf_counter() - start
        except asyncio.TimeoutError:
            runtime_state.record_backend_error("generation_timeout")
            return protocol_error_response(
                status_code=504,
                code="backend_timeout",
                message="generation exceeded gateway timeout",
            )
        except VllmHttpError as exc:
            return _vllm_error_response(exc, runtime_state=runtime_state)
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=elapsed,
                batch_size=1,
            )
            mtp_after = await _mtp_state_for_generation(vllm, served_model_name)
            runtime_state.record_mtp_delta(mtp_metric_delta(mtp_before, mtp_after))
            return JSONResponse(_redact_openai_response(response))
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
            payload, openai_model=served_model_name,
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
            try:
                mtp_before = await _mtp_state_for_generation(vllm, served_model_name)
            except BaseException:
                slot.release()
                raise

            async def event_stream():
                prompt_tokens = 0
                start = time.perf_counter()
                delta_tokens = 0
                usage_tokens: int | None = None
                stream_completed = False
                try:
                    async for event in openai_stream_to_anthropic_events_async(
                        vllm.chat_completion_stream(openai_body),
                        anthropic_model=payload.get("model")
                        or DEFAULT_ANTHROPIC_MODEL_ALIAS,
                        message_id_prefix="msg",
                        prompt_tokens=prompt_tokens,
                    ):
                        if event.get("type") == "content_block_delta":
                            delta_tokens += 1
                        if event.get("type") == "message_delta":
                            usage = event.get("usage") or {}
                            try:
                                usage_tokens = int(usage.get("output_tokens"))
                            except (TypeError, ValueError):
                                pass
                        if event.get("type") == "message_stop":
                            stream_completed = True
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
                    try:
                        if stream_completed or delta_tokens > 0 or usage_tokens is not None:
                            runtime_state.record_generation(
                                generation_tokens=(
                                    usage_tokens
                                    if usage_tokens is not None
                                    else delta_tokens
                                ),
                                generation_seconds=time.perf_counter() - start,
                                batch_size=1,
                            )
                            mtp_after = await _mtp_state_for_generation(
                                vllm,
                                served_model_name,
                            )
                            runtime_state.record_mtp_delta(
                                mtp_metric_delta(mtp_before, mtp_after)
                            )
                    finally:
                        slot.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            mtp_before = await _mtp_state_for_generation(vllm, served_model_name)
            start = time.perf_counter()
            openai_response = await _with_generation_timeout(
                vllm.chat_completion(openai_body),
                server_limits.generation_timeout_seconds,
            )
            elapsed = time.perf_counter() - start
        except asyncio.TimeoutError:
            runtime_state.record_backend_error("generation_timeout")
            return protocol_error_response(
                status_code=504,
                code="backend_timeout",
                message="generation exceeded gateway timeout",
                protocol="anthropic",
            )
        except VllmHttpError as exc:
            return _vllm_error_response(
                exc,
                runtime_state=runtime_state,
                protocol="anthropic",
            )
        else:
            runtime_state.record_generation(
                generation_tokens=int(
                    (openai_response.get("usage") or {}).get("completion_tokens") or 0
                ),
                generation_seconds=elapsed,
                batch_size=1,
            )
            mtp_after = await _mtp_state_for_generation(vllm, served_model_name)
            runtime_state.record_mtp_delta(mtp_metric_delta(mtp_before, mtp_after))
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
        if not _alias_known(payload.get("model"), aliases):
            return protocol_error_response(
                status_code=404,
                code="model_not_found",
                message=f"model {payload.get('model')!r} is not available",
                protocol="anthropic",
            )
        openai_body = anthropic_request_to_openai(
            payload,
            openai_model=served_model_name,
        )
        try:
            tokenized = await vllm.tokenize(
                {
                    "model": served_model_name,
                    "messages": openai_body.get("messages", []),
                }
            )
            input_tokens = int(tokenized["count"])
        except VllmHttpError as exc:
            return _vllm_error_response(
                exc,
                runtime_state=runtime_state,
                protocol="anthropic",
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            runtime_state.record_backend_error("tokenizer_unavailable")
            return protocol_error_response(
                status_code=503,
                code="backend_unavailable",
                message="vllm tokenizer endpoint is unavailable",
                protocol="anthropic",
            )
        return JSONResponse(
            {"input_tokens": max(0, input_tokens)},
            headers={"X-Gemma4-MTP-Token-Counting": "backend_tokenizer"},
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


async def _observed_backend_config(
    vllm: VllmClient,
    profile: ModelProfile,
    *,
    served_model_name: str | None,
    vllm_version: str | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    active_backend_pid: int | None = None,
    active_backend_argv: list[str] | None = None,
    vllm_log_path: Path | None = None,
) -> dict[str, Any]:
    observed = merge_observed_config(
        default_observed_config(),
        observed_config_from_version(vllm_version),
        observed_config_from_runtime_evidence(
            runtime_manifest=runtime_manifest,
            runtime_manifest_path=runtime_manifest_path,
            active_backend_pid=active_backend_pid,
            active_backend_argv=active_backend_argv,
            vllm_base_url=vllm.base_url,
        ),
    )
    try:
        models_body = await _vllm_readiness_probe(vllm.list_models())
    except (asyncio.TimeoutError, VllmHttpError, httpx.HTTPError):
        pass
    else:
        observed = merge_observed_config(
            observed,
            observed_config_from_models(
                models_body,
                target_model=profile.target,
                served_model_name=served_model_name,
            ),
        )
    try:
        observed = merge_observed_config(
            observed,
            observed_config_from_metrics(
                await _vllm_readiness_probe(vllm.metrics_text()),
                model_name=served_model_name,
                log_text=read_text_tail(vllm_log_path),
            ),
        )
    except (asyncio.TimeoutError, VllmHttpError, httpx.HTTPError):
        observed = merge_observed_config(
            observed,
            observed_config_from_metrics(
                "",
                model_name=served_model_name,
                log_text=read_text_tail(vllm_log_path),
            ),
        )
    return observed


async def _mtp_state_for_generation(
    vllm: VllmClient,
    served_model_name: str | None,
    *,
    timeout_seconds: float = MTP_METRICS_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        metrics_text = await asyncio.wait_for(
            vllm.metrics_text(),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, VllmHttpError, httpx.HTTPError):
        metrics_text = ""
    observed = observed_config_from_metrics(
        metrics_text,
        model_name=served_model_name,
    )
    mtp = observed.get("mtp")
    return mtp if isinstance(mtp, dict) else {}


async def _with_generation_timeout(coro, timeout_seconds: float | None):
    if timeout_seconds is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


async def _vllm_readiness_probe(coro):
    return await asyncio.wait_for(
        coro,
        timeout=VLLM_READINESS_PROBE_TIMEOUT_SECONDS,
    )


def _vllm_error_response(
    exc: VllmHttpError,
    *,
    runtime_state: RuntimeState,
    protocol: str = "openai",
) -> JSONResponse:
    code, message = _upstream_error_details(exc)
    if 400 <= exc.status_code < 500:
        return protocol_error_response(
            status_code=exc.status_code,
            code=code,
            message=message,
            protocol=protocol,
        )
    runtime_state.record_backend_error("vllm_http_error")
    return protocol_error_response(
        status_code=503,
        code="backend_unavailable",
        message=f"vllm returned {exc.status_code}",
        protocol=protocol,
    )


def _upstream_error_details(exc: VllmHttpError) -> tuple[str, str]:
    try:
        body = json.loads(str(exc))
    except json.JSONDecodeError:
        return "upstream_error", str(exc)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return "upstream_error", str(exc)
    code = error.get("code") or error.get("type") or "upstream_error"
    message = error.get("message") or str(exc)
    return str(code), str(message)


def _stream_usage_tokens(chunk: dict[str, Any], current: int | None) -> int | None:
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return current
    try:
        return int(usage.get("completion_tokens"))
    except (TypeError, ValueError):
        return current


def _chunk_has_content_delta(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and delta.get("content"):
            return True
    return False


async def _probe_vllm(vllm: VllmClient) -> dict[str, Any]:
    try:
        body = await _vllm_readiness_probe(vllm.health())
        if isinstance(body, dict) and body.get("status") == "ok":
            return {"status": "ok"}
        return {"status": "degraded", "raw": body}
    except VllmHttpError as exc:
        return {"status": "unreachable", "http_status": exc.status_code}
    except asyncio.TimeoutError:
        return {"status": "unreachable", "error": "timeout"}
    except httpx.HTTPError as exc:
        return {"status": "unreachable", "error": str(exc)}


async def _probe_vllm_with_version(vllm: VllmClient) -> tuple[dict[str, Any], bool]:
    vllm_status = await _probe_vllm(vllm)
    if vllm_status.get("status") == "ok":
        try:
            version_body = await _vllm_readiness_probe(vllm.version())
            vllm_status["version"] = version_body.get("version")
        except (asyncio.TimeoutError, VllmHttpError, httpx.HTTPError):
            vllm_status["version"] = None

    version_ok = version_at_least(
        vllm_status.get("version"),
        REQUIRED_VLLM_MIN_VERSION,
    )
    return vllm_status, version_ok
