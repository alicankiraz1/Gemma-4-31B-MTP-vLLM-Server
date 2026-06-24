from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
import typer
import uvicorn

from gemma4_mtp_vllm import __version__
from gemma4_mtp_vllm.benchmarking import (
    BenchmarkEndpointResult,
    BenchmarkObservation,
    BenchmarkSummary,
    metric_summary,
    deterministic_parity,
    percentile,
    speedup,
)
from gemma4_mtp_vllm.doctor import build_report
from gemma4_mtp_vllm.launch import (
    build_vllm_serve_args,
    resolve_vllm_executable,
    write_launch_manifest,
)
from gemma4_mtp_vllm.mtp_metrics import mtp_metric_delta, parse_mtp_metrics
from gemma4_mtp_vllm.profiles import (
    ModelProfile,
    ProfileSet,
    load_profiles,
    resolve_profile,
)
from gemma4_mtp_vllm.server.app import DEFAULT_MODEL_ALIAS
from gemma4_mtp_vllm.server.app import create_app
from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key
from gemma4_mtp_vllm.server.limits import ServerLimits

app = typer.Typer(add_completion=False, help="Gemma 4 31B MTP vLLM sidecar gateway")

P1_001_CONTROL_PROFILE = "tp2_2x32_fp8_gpuonly"
P1_001_CANDIDATE_PROFILE = "tp2_2x32_fp8_gpuonly_cuda_graph"
P1_001_REQUIRED_OUTPUT_TOKEN_TARGETS = (64, 256, 512, 1024)
P1_001_EXPECTED_GPU_COUNT = 2
P1_001_MIN_SOAK_SECONDS = 3600.0
P1_001_ACCEPTANCE_RATE_MARGIN = -0.01
P1_001_MEAN_ACCEPTANCE_LENGTH_MARGIN = -0.05
BENCHMARK_PROTOCOL_VERSION = 3


def _profile_set() -> ProfileSet:
    return load_profiles()


def _mock_transport():
    """Test-only hook overridden in tests when VLLM_MTP_TRANSPORT_MOCK=1."""
    return None


def _build_transport() -> httpx.BaseTransport | None:
    if os.environ.get("VLLM_MTP_TRANSPORT_MOCK") == "1":
        return _mock_transport()
    return None


def _request_body(
    profile: ModelProfile,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = 1,
) -> dict:
    body = {
        "model": DEFAULT_MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _http_client(
    base_url: str,
    *,
    timeout: httpx.Timeout | None = None,
    limits: httpx.Limits | None = None,
) -> httpx.AsyncClient:
    timeout = timeout or _vllm_http_timeout()
    limits = limits or _vllm_http_limits()
    transport = _build_transport()
    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "timeout": timeout,
        "limits": limits,
        "http1": True,
        "http2": False,
    }
    if transport is not None:
        kwargs["transport"] = transport
    return httpx.AsyncClient(**kwargs)


def _vllm_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)


def _vllm_http_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=20,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    )


def _timeout_metadata(timeout: httpx.Timeout) -> dict[str, float | None]:
    return {
        "connect": timeout.connect,
        "read": timeout.read,
        "write": timeout.write,
        "pool": timeout.pool,
    }


def _limits_metadata(limits: httpx.Limits) -> dict[str, float | int | None]:
    return {
        "max_connections": limits.max_connections,
        "max_keepalive_connections": limits.max_keepalive_connections,
        "keepalive_expiry": limits.keepalive_expiry,
    }


class BenchmarkHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        http: httpx.AsyncClient,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
    ) -> None:
        self.base_url = base_url
        self._http = http
        self._timeout = timeout
        self._limits = limits
        self._request_count = 0
        self._connection_use_counts: dict[int, int] = {}
        self._connection_reuse_count = 0
        self._connection_reuse_observable = False
        self._last_http_version: str | None = None

    @classmethod
    def create(cls, base_url: str) -> "BenchmarkHttpClient":
        timeout = _vllm_http_timeout()
        limits = _vllm_http_limits()
        return cls(
            base_url=base_url,
            http=_http_client(base_url, timeout=timeout, limits=limits),
            timeout=timeout,
            limits=limits,
        )

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        json_body: dict,
    ) -> AsyncIterator[httpx.Response]:
        self._request_count += 1
        async with self._http.stream(method, url, json=json_body) as response:
            self._record_response(response)
            yield response

    async def get(self, url: str) -> httpx.Response:
        self._request_count += 1
        response = await self._http.get(url)
        self._record_response(response)
        return response

    async def post(self, url: str, *, json_body: dict) -> httpx.Response:
        self._request_count += 1
        response = await self._http.post(url, json=json_body)
        self._record_response(response)
        return response

    async def aclose(self) -> None:
        await self._http.aclose()

    def metadata(self, response: httpx.Response | None = None) -> dict[str, object]:
        http_version = response.http_version if response is not None else None
        if http_version is None:
            http_version = self._last_http_version
        return {
            "http_version": http_version,
            "connection_reuse_count": (
                self._connection_reuse_count
                if self._connection_reuse_observable
                else None
            ),
            "connection_reuse_observable": self._connection_reuse_observable,
            "client_request_count": self._request_count,
            "client_reuse_count": max(self._request_count - 1, 0),
            "timeout": _timeout_metadata(self._timeout),
            "limits": _limits_metadata(self._limits),
        }

    def _record_response(self, response: httpx.Response) -> None:
        self._last_http_version = response.http_version
        network_stream = response.extensions.get("network_stream")
        if network_stream is None:
            return
        self._connection_reuse_observable = True
        key = id(network_stream)
        prior_uses = self._connection_use_counts.get(key, 0)
        if prior_uses > 0:
            self._connection_reuse_count += 1
        self._connection_use_counts[key] = prior_uses + 1


class BenchmarkClientPool:
    def __init__(self) -> None:
        self._clients: dict[str, BenchmarkHttpClient] = {}

    async def __aenter__(self) -> "BenchmarkClientPool":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        try:
            await self.aclose()
        except BaseException as close_exc:
            if exc is not None:
                _raise_benchmark_close_errors(
                    [exc, *_benchmark_close_error_list(close_exc)]
                )
            raise

    def client_for(self, base_url: str) -> BenchmarkHttpClient:
        client = self._clients.get(base_url)
        if client is None:
            client = BenchmarkHttpClient.create(base_url)
            self._clients[base_url] = client
        return client

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        close_errors: list[BaseException] = []
        for client in clients:
            try:
                await client.aclose()
            except BaseException as exc:
                close_errors.append(exc)
        _raise_benchmark_close_errors(close_errors)


class BenchmarkClientCloseError(Exception):
    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in self.errors
        )
        super().__init__(f"benchmark client close failed: {details}")


def _benchmark_close_error_list(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BenchmarkClientCloseError):
        return exc.errors
    return [exc]


def _raise_benchmark_close_errors(errors: list[BaseException]) -> None:
    if not errors:
        return
    cancellation = next(
        (error for error in errors if isinstance(error, asyncio.CancelledError)),
        None,
    )
    if cancellation is not None:
        raise cancellation
    if len(errors) == 1:
        raise errors[0]
    raise BenchmarkClientCloseError(errors)


async def _measure(
    base_url: str,
    body: dict,
    *,
    client: BenchmarkHttpClient | None = None,
    capture_raw_stream: bool = False,
) -> tuple[str, BenchmarkEndpointResult]:
    if client is None:
        async with BenchmarkClientPool() as pool:
            return await _measure(
                base_url,
                body,
                client=pool.client_for(base_url),
                capture_raw_stream=capture_raw_stream,
            )

    start_ns = _now_ns()
    generated_token_chunk_timestamps_ns: list[int] = []
    visible_content_chunk_timestamps_ns: list[int] = []
    chunk_timestamps_ns: list[int] = []
    raw_output_token_ids: list[int] = []
    output_parts: list[str] = []
    reasoning_parts: list[str] = []
    token_chunk_events: list[dict[str, object]] = []
    stream_parse_errors: list[dict[str, object]] = []
    raw_stream_payload_candidates: list[str] = []
    raw_stream_chunk_candidates: list[dict[str, Any]] = []
    raw_rejected_event_indices: list[int] = []
    raw_rejection_reasons: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    transport_metadata: dict[str, object] | None = None
    event_index = 0
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json_body=body,
    ) as response:
        transport_metadata = client.metadata(response)
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if raw == "[DONE]":
                break
            event_index += 1
            timestamp_ns = _now_ns()
            chunk_timestamps_ns.append(timestamp_ns)
            if capture_raw_stream:
                rejection_reasons = _raw_stream_rejection_reasons(raw)
                if rejection_reasons:
                    raw_rejected_event_indices.append(event_index)
                    raw_rejection_reasons.extend(rejection_reasons)
                raw_stream_payload_candidates.append(raw)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                stream_parse_errors.append(
                    _stream_parse_error_event(
                        raw=raw,
                        event_index=event_index,
                        timestamp_ns=timestamp_ns,
                        error_type=type(exc).__name__,
                    )
                )
                continue
            if not isinstance(payload, dict):
                stream_parse_errors.append(
                    _stream_parse_error_event(
                        raw=raw,
                        event_index=event_index,
                        timestamp_ns=timestamp_ns,
                        error_type="InvalidPayloadType",
                    )
                )
                continue
            if capture_raw_stream:
                raw_stream_chunk_candidates.append(payload)
            event = _stream_chunk_event(
                payload,
                event_index=event_index,
                timestamp_ns=timestamp_ns,
                raw=raw,
            )
            token_chunk_events.append(event)
            delta_value = event.get("delta")
            delta = delta_value if isinstance(delta_value, dict) else {}
            token_ids_value = delta.get("token_ids")
            chunk_token_ids = (
                token_ids_value if isinstance(token_ids_value, list) else []
            )
            usage = event["usage"]
            if isinstance(usage, dict):
                prompt_token_value = usage.get("prompt_tokens")
                completion_token_value = usage.get("completion_tokens")
                if isinstance(prompt_token_value, int):
                    prompt_tokens = prompt_token_value
                if isinstance(completion_token_value, int):
                    completion_tokens = completion_token_value
            if chunk_token_ids:
                raw_output_token_ids.extend(chunk_token_ids)
                generated_token_chunk_timestamps_ns.append(timestamp_ns)
            reasoning = delta.get("reasoning")
            reasoning_content = delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(str(reasoning))
            elif reasoning_content:
                reasoning_parts.append(str(reasoning_content))
            content = delta.get("content")
            if content:
                if not visible_content_chunk_timestamps_ns:
                    visible_content_chunk_timestamps_ns.append(timestamp_ns)
                output_parts.append(str(content))
    end_ns = _now_ns()
    text = "".join(output_parts)
    reasoning_text = "".join(reasoning_parts)
    if not raw_output_token_ids and completion_tokens is None and text:
        completion_tokens = await _count_output_tokens(base_url, text, client=client)
    observed_completion_tokens: int | None = completion_tokens
    token_count_status = (
        "malformed_stream" if stream_parse_errors else "raw_token_ids_missing"
    )
    timing_evidence_valid = False
    if raw_output_token_ids:
        raw_count = len(raw_output_token_ids)
        if stream_parse_errors:
            observed_completion_tokens = completion_tokens
        elif completion_tokens is None:
            token_count_status = "usage_missing"
            observed_completion_tokens = None
        elif completion_tokens == raw_count:
            token_count_status = "matched"
            timing_evidence_valid = True
            observed_completion_tokens = raw_count
        else:
            token_count_status = "usage_mismatch"
            observed_completion_tokens = completion_tokens
    token_count_diagnostics = {
        "raw_output_token_count": len(raw_output_token_ids),
        "usage_completion_tokens": completion_tokens,
        "stream_parse_error_count": len(stream_parse_errors),
    }
    raw_stream_payloads: list[str] | None = None
    raw_stream_chunks: list[dict[str, Any]] | None = None
    raw_stream_capture_diagnostics: dict[str, object] | None = None
    if not capture_raw_stream:
        raw_stream_capture_status = "disabled"
    elif raw_rejected_event_indices:
        raw_stream_capture_status = "rejected_by_sanitizer"
        raw_stream_capture_diagnostics = {
            "rejected_event_indices": raw_rejected_event_indices,
            "reasons": _dedupe(raw_rejection_reasons),
        }
    else:
        raw_stream_capture_status = "captured"
        raw_stream_payloads = raw_stream_payload_candidates
        raw_stream_chunks = raw_stream_chunk_candidates
    total_latency_ms = _elapsed_ms(start_ns, end_ns)
    generated_ttft_ms = (
        _elapsed_ms(start_ns, generated_token_chunk_timestamps_ns[0])
        if timing_evidence_valid and generated_token_chunk_timestamps_ns
        else None
    )
    visible_content_ttft_ms = (
        _elapsed_ms(start_ns, visible_content_chunk_timestamps_ns[0])
        if timing_evidence_valid and visible_content_chunk_timestamps_ns
        else None
    )
    chunk_intervals_ms = (
        [
            _elapsed_ms(left, right)
            for left, right in zip(chunk_timestamps_ns, chunk_timestamps_ns[1:])
        ]
        if len(chunk_timestamps_ns) > 1
        else []
    )
    elapsed_seconds = max((end_ns - start_ns) / 1_000_000_000.0, 0.0)
    throughput_token_count = (
        observed_completion_tokens
        if not stream_parse_errors
        and (timing_evidence_valid or not raw_output_token_ids)
        else None
    )
    throughput = (
        throughput_token_count / elapsed_seconds
        if elapsed_seconds > 0
        and throughput_token_count is not None
        and throughput_token_count > 0
        else None
    )
    tpot_ms = (
        _elapsed_ms(
            generated_token_chunk_timestamps_ns[0],
            generated_token_chunk_timestamps_ns[-1],
        )
        / (len(raw_output_token_ids) - 1)
        if timing_evidence_valid
        and len(raw_output_token_ids) > 1
        and len(generated_token_chunk_timestamps_ns) > 1
        else None
    )
    result = BenchmarkEndpointResult(
        e2e_output_tokens_per_second=throughput,
        ttft_ms=generated_ttft_ms,
        tpot_ms=tpot_ms,
        inter_token_latency_ms_p50=None,
        inter_token_latency_ms_p95=None,
        total_latency_ms=total_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=observed_completion_tokens,
        raw_output_token_ids=raw_output_token_ids or None,
        reasoning_text=reasoning_text,
        visible_content=text,
        generated_ttft_ms=generated_ttft_ms,
        visible_content_ttft_ms=visible_content_ttft_ms,
        token_chunk_events=token_chunk_events or None,
        chunk_timestamps_ns=chunk_timestamps_ns or None,
        token_timing_basis=(
            "raw_token_ids_chunk_arrival" if raw_output_token_ids else "unavailable"
        ),
        tpot_basis=(
            "chunk_arrival_approximation"
            if tpot_ms is not None
            else "unavailable"
        ),
        itl_basis=(
            "not_reported_chunk_interval_only" if raw_output_token_ids else "unavailable"
        ),
        stream_chunk_interval_ms_p50=percentile(chunk_intervals_ms, 50),
        stream_chunk_interval_ms_p95=percentile(chunk_intervals_ms, 95),
        timing_evidence_valid=timing_evidence_valid,
        token_count_validation_status=token_count_status,
        token_count_diagnostics=token_count_diagnostics,
        stream_parse_errors=stream_parse_errors or None,
        raw_stream_chunks=raw_stream_chunks,
        raw_stream_payloads=raw_stream_payloads,
        raw_stream_capture_status=raw_stream_capture_status,
        raw_stream_capture_diagnostics=raw_stream_capture_diagnostics,
        stream_interval_control="unavailable",
        transport_metadata=transport_metadata,
    )
    return text, result


def _now_ns() -> int:
    return int(time.perf_counter() * 1_000_000_000)


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def _stream_chunk_event(
    payload: dict[str, Any],
    *,
    event_index: int,
    timestamp_ns: int,
    raw: str,
) -> dict[str, object]:
    token_ids = _stream_chunk_token_ids(payload)
    reasoning_parts: list[str] = []
    reasoning_content_parts: list[str] = []
    content_parts: list[str] = []
    finish_reason: str | None = None
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if finish_reason is None and choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason"))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            reasoning = _coerce_delta_text(delta.get("reasoning"))
            reasoning_content = _coerce_delta_text(delta.get("reasoning_content"))
            content = _coerce_delta_text(delta.get("content"))
            if reasoning:
                reasoning_parts.append(reasoning)
            if reasoning_content:
                reasoning_content_parts.append(reasoning_content)
            if content:
                content_parts.append(content)
    return {
        "event_index": event_index,
        "timestamp_ns": timestamp_ns,
        "delta": {
            "token_ids": token_ids,
            "reasoning": "".join(reasoning_parts),
            "reasoning_content": "".join(reasoning_content_parts),
            "content": "".join(content_parts),
        },
        "usage": _sanitize_usage(payload.get("usage")),
        "finish_reason": finish_reason,
        "token_count": len(token_ids),
        "multiple_token_ids": len(token_ids) > 1,
        "payload_sha256": _payload_sha256(raw),
    }


def _coerce_delta_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sanitize_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        token_count = value.get(key)
        if isinstance(token_count, int) and not isinstance(token_count, bool):
            usage[key] = token_count
    return usage or None


def _stream_parse_error_event(
    *,
    raw: str,
    event_index: int,
    timestamp_ns: int,
    error_type: str,
) -> dict[str, object]:
    return {
        "event_index": event_index,
        "timestamp_ns": timestamp_ns,
        "error": error_type,
        "payload_sha256": _payload_sha256(raw),
        "payload_bytes": len(raw.encode("utf-8")),
    }


def _payload_sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_RAW_STREAM_SECRET_PATTERNS = (
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]+"),
    re.compile(
        r"(?i)\"(?:api[_-]?key|authorization|password|secret|"
        r"access[_-]?token|refresh[_-]?token|private[_-]?key|"
        r"client[_-]?secret)\"\s*:"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]+"),
)
_RAW_STREAM_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


def _raw_stream_rejection_reasons(raw: str) -> list[str]:
    reasons: list[str] = []
    if any(pattern.search(raw) for pattern in _RAW_STREAM_SECRET_PATTERNS):
        reasons.append("secret_pattern")
    if _RAW_STREAM_EMAIL_PATTERN.search(raw):
        reasons.append("pii_pattern")
    return reasons


def _coerce_token_ids(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    tokens: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            tokens.append(item)
    return tokens


def _stream_chunk_token_ids(payload: dict[str, object]) -> list[int]:
    tokens: list[int] = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                direct = _coerce_token_ids(delta.get("token_ids"))
                if direct:
                    tokens.extend(direct)
                    continue
                tokens.extend(_coerce_token_ids(delta.get("reasoning_token_ids")))
                tokens.extend(_coerce_token_ids(delta.get("content_token_ids")))
            tokens.extend(_coerce_token_ids(choice.get("token_ids")))
    if not tokens:
        tokens.extend(_coerce_token_ids(payload.get("token_ids")))
    return tokens


async def _fetch_mtp_metrics(
    base_url: str,
    *,
    client: BenchmarkHttpClient | None = None,
) -> dict:
    return parse_mtp_metrics(await _fetch_metrics_text(base_url, client=client))


async def _fetch_metrics_text(
    base_url: str,
    *,
    client: BenchmarkHttpClient | None = None,
) -> str:
    if client is None:
        async with BenchmarkClientPool() as pool:
            return await _fetch_metrics_text(
                base_url,
                client=pool.client_for(base_url),
            )
    try:
        response = await client.get("/metrics")
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


async def _tokenize_visible_text(
    base_url: str,
    text: str,
    *,
    client: BenchmarkHttpClient | None = None,
) -> dict | None:
    if client is None:
        async with BenchmarkClientPool() as pool:
            return await _tokenize_visible_text(
                base_url,
                text,
                client=pool.client_for(base_url),
            )
    try:
        response = await client.post(
            "/tokenize",
            json_body={
                "model": DEFAULT_MODEL_ALIAS,
                "prompt": text,
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _tokens_from_tokenize_payload(payload: dict | None) -> list[int] | None:
    if payload is None:
        return None
    tokens = payload.get("tokens")
    if isinstance(tokens, list) and all(isinstance(token, int) for token in tokens):
        return tokens
    return None


def _count_from_tokenize_payload(payload: dict | None) -> int | None:
    if payload is None:
        return None
    tokens = _tokens_from_tokenize_payload(payload)
    if tokens is not None:
        return len(tokens)
    count = payload.get("count")
    if isinstance(count, int) and count >= 0:
        return count
    return None


async def _count_output_tokens(
    base_url: str,
    text: str,
    *,
    client: BenchmarkHttpClient | None = None,
) -> int | None:
    return _count_from_tokenize_payload(
        await _tokenize_visible_text(base_url, text, client=client)
    )


async def _tokenize_output(
    base_url: str,
    text: str,
    *,
    client: BenchmarkHttpClient | None = None,
) -> list[int] | None:
    return _tokens_from_tokenize_payload(
        await _tokenize_visible_text(base_url, text, client=client)
    )


async def _deterministic_parity_for_outputs(
    *,
    baseline_url: str,
    mtp_url: str,
    baseline_text: str,
    mtp_text: str,
    baseline_client: BenchmarkHttpClient | None = None,
    mtp_client: BenchmarkHttpClient | None = None,
) -> tuple[bool | None, str]:
    baseline_tokens = await _tokenize_output(
        baseline_url,
        baseline_text,
        client=baseline_client,
    )
    mtp_tokens = await _tokenize_output(
        mtp_url,
        mtp_text,
        client=mtp_client,
    )
    if baseline_tokens is not None and mtp_tokens is not None:
        return baseline_tokens == mtp_tokens, "token"
    return (
        deterministic_parity(
            baseline_text,
            mtp_text,
            temperature=0.0,
            top_p=1.0,
        ),
        "text",
    )


async def _single_bench(
    *,
    profile: ModelProfile,
    prompt: str,
    max_tokens: int,
    mtp_url: str,
    baseline_url: str,
    runs: int,
    warmup_runs: int,
    client_pool: BenchmarkClientPool | None = None,
    capture_raw_stream: bool = False,
) -> list[BenchmarkObservation]:
    if client_pool is None:
        async with BenchmarkClientPool() as pool:
            return await _single_bench(
                profile=profile,
                prompt=prompt,
                max_tokens=max_tokens,
                mtp_url=mtp_url,
                baseline_url=baseline_url,
                runs=runs,
                warmup_runs=warmup_runs,
                client_pool=pool,
                capture_raw_stream=capture_raw_stream,
            )

    mtp_client = client_pool.client_for(mtp_url)
    baseline_client = client_pool.client_for(baseline_url)
    body = _request_body(
        profile,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=1,
    )

    for _ in range(warmup_runs):
        await _measure(mtp_url, body, client=mtp_client)
        await _measure(baseline_url, body, client=baseline_client)

    observations: list[BenchmarkObservation] = []
    for idx in range(1, runs + 1):
        if idx % 2:
            mtp_metrics_before = await _fetch_mtp_metrics(
                mtp_url,
                client=mtp_client,
            )
            mtp_text, mtp_result = await _measure(
                mtp_url,
                body,
                client=mtp_client,
                capture_raw_stream=capture_raw_stream,
            )
            mtp_metrics_after = await _fetch_mtp_metrics(
                mtp_url,
                client=mtp_client,
            )
            no_text, baseline_result = await _measure(
                baseline_url,
                body,
                client=baseline_client,
                capture_raw_stream=capture_raw_stream,
            )
        else:
            no_text, baseline_result = await _measure(
                baseline_url,
                body,
                client=baseline_client,
                capture_raw_stream=capture_raw_stream,
            )
            mtp_metrics_before = await _fetch_mtp_metrics(
                mtp_url,
                client=mtp_client,
            )
            mtp_text, mtp_result = await _measure(
                mtp_url,
                body,
                client=mtp_client,
                capture_raw_stream=capture_raw_stream,
            )
            mtp_metrics_after = await _fetch_mtp_metrics(
                mtp_url,
                client=mtp_client,
            )
        parity, parity_basis = await _deterministic_parity_for_outputs(
            baseline_url=baseline_url,
            mtp_url=mtp_url,
            baseline_text=no_text,
            mtp_text=mtp_text,
            baseline_client=baseline_client,
            mtp_client=mtp_client,
        )
        observations.append(
            BenchmarkObservation(
                index=idx,
                baseline=baseline_result,
                mtp=mtp_result,
                speedup=speedup(
                    baseline_result.e2e_output_tokens_per_second,
                    mtp_result.e2e_output_tokens_per_second,
                ),
                deterministic_parity=parity,
                parity_basis=parity_basis,
                parity_failure=parity is False,
                mtp_metrics_before=mtp_metrics_before,
                mtp_metrics_after=mtp_metrics_after,
                mtp_metrics_delta=mtp_metric_delta(
                    mtp_metrics_before,
                    mtp_metrics_after,
                ),
            )
        )
    return observations


@app.command()
def doctor(
    profile: str = typer.Option("safe80", "--profile"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    runtime_manifest_path: Optional[Path] = typer.Option(
        Path("logs/vllm-launch-manifest.json"),
        "--runtime-manifest-path",
    ),
) -> None:
    profile_set = _profile_set()
    selected = resolve_profile(profile, profile_set)
    transport = _build_transport()
    report = asyncio.run(
        build_report(
            profile=selected,
            vllm_base_url=vllm_base_url,
            transport=transport,
            served_model_name=DEFAULT_MODEL_ALIAS,
            runtime_manifest_path=runtime_manifest_path,
        )
    )
    # Emit single-line JSON so the test seam (splitlines()[-1]) yields a
    # parseable payload; multi-line indented output would leave the test
    # parsing just the closing brace.
    typer.echo(json.dumps(report))


@app.command()
def launch(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    print_only: bool = typer.Option(False, "--print-only"),
    no_mtp: bool = typer.Option(False, "--no-mtp"),
    allow_public_vllm: bool = typer.Option(False, "--allow-public-vllm"),
    manifest_path: Optional[Path] = typer.Option(
        Path("logs/vllm-launch-manifest.json"),
        "--manifest-path",
    ),
) -> None:
    if bind_host_requires_api_key(host) and not allow_public_vllm:
        typer.echo(
            "raw vLLM should stay on loopback; pass --allow-public-vllm to expose it",
            err=True,
        )
        raise typer.Exit(code=1)

    selected = resolve_profile(profile, _profile_set())
    args = build_vllm_serve_args(
        profile=selected,
        host=host,
        port=port,
        enable_mtp=not no_mtp,
        served_model_name=DEFAULT_MODEL_ALIAS,
    )
    if print_only:
        typer.echo(shlex.join(args))
        return
    if manifest_path is not None:
        write_launch_manifest(
            path=manifest_path,
            profile=selected,
            argv=args,
            enable_mtp=not no_mtp,
            served_model_name=DEFAULT_MODEL_ALIAS,
        )
    os.execvp(resolve_vllm_executable(args[0]), args)


@app.command()
def serve(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    max_body_mb: float = typer.Option(2.0, "--max-body-mb"),
    max_output_tokens: Optional[int] = typer.Option(None, "--max-output-tokens"),
    max_queue_size: int = typer.Option(8, "--max-queue-size"),
    rate_limit_rpm: int = typer.Option(30, "--rate-limit-rpm"),
    generation_timeout_seconds: Optional[float] = typer.Option(
        900.0,
        "--generation-timeout-seconds",
    ),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    runtime_manifest_path: Optional[Path] = typer.Option(
        Path("logs/vllm-launch-manifest.json"),
        "--runtime-manifest-path",
    ),
    cors_origin: list[str] = typer.Option([], "--cors-origin"),
) -> None:
    if bind_host_requires_api_key(host) and not api_key:
        typer.echo(f"host {host} requires --api-key", err=True)
        raise typer.Exit(code=1)

    selected = resolve_profile(profile, _profile_set())
    limits = ServerLimits(
        max_body_bytes=int(max_body_mb * 1024 * 1024),
        max_output_tokens=max_output_tokens or selected.max_output_tokens,
        max_queue_size=max_queue_size,
        rate_limit_rpm=rate_limit_rpm,
        cors_origins=tuple(cors_origin),
        generation_timeout_seconds=generation_timeout_seconds,
    )
    fastapi_app = create_app(
        profile_name=selected.name,
        bind_host=host,
        api_key=api_key,
        limits=limits,
        vllm_base_url=vllm_base_url,
        runtime_manifest_path=runtime_manifest_path,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def generate(
    prompt: str = typer.Argument(...),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    no_mtp: bool = typer.Option(
        False,
        "--no-mtp",
        help="Requires a separate vLLM launch without speculative config.",
    ),
) -> None:
    """One-shot generation via the configured vLLM server."""
    if no_mtp:
        typer.echo(
            "--no-mtp requires launching a separate vLLM process without "
            "--speculative-config; see `vllm-mtp bench` for paired runs.",
            err=True,
        )
        raise typer.Exit(code=2)

    selected = resolve_profile(profile, _profile_set())

    async def run() -> dict:
        async with httpx.AsyncClient(
            base_url=vllm_base_url,
            timeout=_vllm_http_timeout(),
        ) as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ALIAS,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            response.raise_for_status()
            return response.json()

    payload = asyncio.run(run())
    text = payload["choices"][0]["message"]["content"]
    typer.echo(text)


@app.command()
def bench(
    prompt: str = typer.Option(..., "--prompt"),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens", "--output-token-target"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    runs: int = typer.Option(10, "--runs"),
    warmup_runs: int = typer.Option(2, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
    artifact_root: Optional[Path] = typer.Option(None, "--artifact-root"),
    artifact_id: Optional[str] = typer.Option(None, "--artifact-id"),
    runtime_manifest_path: Optional[Path] = typer.Option(
        None,
        "--runtime-manifest-path",
    ),
    capture_raw_stream: bool = typer.Option(False, "--capture-raw-stream"),
) -> None:
    """Compare MTP vs baseline vLLM endpoints for a single prompt."""
    selected = resolve_profile(profile, _profile_set())

    async def run() -> tuple[str, list[BenchmarkObservation], str]:
        async with BenchmarkClientPool() as pool:
            mtp_client = pool.client_for(mtp_url)
            metrics_before = (
                await _fetch_metrics_text(mtp_url, client=mtp_client)
                if artifact_root is not None
                else ""
            )
            benchmark_observations = await _single_bench(
                profile=selected,
                prompt=prompt,
                max_tokens=max_tokens,
                mtp_url=mtp_url,
                baseline_url=baseline_url,
                runs=runs,
                warmup_runs=warmup_runs,
                client_pool=pool,
                capture_raw_stream=capture_raw_stream,
            )
            metrics_after = (
                await _fetch_metrics_text(mtp_url, client=mtp_client)
                if artifact_root is not None
                else ""
            )
            return metrics_before, benchmark_observations, metrics_after

    metrics_before_text, observations, metrics_after_text = asyncio.run(run())
    summary = BenchmarkSummary(
        profile=selected.name,
        prompt_name="default",
        prompt=prompt,
        output_token_target=max_tokens,
        num_speculative_tokens=selected.num_speculative_tokens,
        observations=observations,
    )
    payload = summary.to_dict()
    request_body = _request_body(
        selected,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=1,
    )
    if artifact_root is not None:
        _write_benchmark_artifacts(
            artifact_root=artifact_root,
            artifact_id=artifact_id,
            profile=selected,
            summary_payload=payload,
            request_body=request_body,
            mtp_url=mtp_url,
            baseline_url=baseline_url,
            metrics_before_text=metrics_before_text,
            metrics_after_text=metrics_after_text,
            runtime_manifest_path=runtime_manifest_path,
        )
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if json_output:
        Path(json_output).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@app.command("bench-single")
def bench_single(
    url: str = typer.Option(..., "--url"),
    label: str = typer.Option("endpoint", "--label"),
    profile: str = typer.Option("safe80", "--profile"),
    prompt: list[str] = typer.Option([], "--prompt"),
    output_token_target: list[int] = typer.Option([], "--output-token-target"),
    runs: int = typer.Option(10, "--runs"),
    warmup_runs: int = typer.Option(2, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
    capture_raw_stream: bool = typer.Option(False, "--capture-raw-stream"),
) -> None:
    """Measure one runtime endpoint for sequential A/B experiments."""
    if not prompt:
        typer.echo("at least one --prompt required", err=True)
        raise typer.Exit(code=2)
    if runs <= 0:
        typer.echo("--runs must be positive", err=True)
        raise typer.Exit(code=2)
    if warmup_runs < 0:
        typer.echo("--warmup-runs must be non-negative", err=True)
        raise typer.Exit(code=2)
    output_token_targets = output_token_target or [64]
    if any(target <= 0 for target in output_token_targets):
        typer.echo("--output-token-target must be positive", err=True)
        raise typer.Exit(code=2)

    output_path = Path(json_output) if json_output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = resolve_profile(profile, _profile_set())
    groups: list[dict[str, object]] = []
    payload = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "benchmark_kind": "single_endpoint_runtime",
        "status": "in_progress",
        "label": label,
        "profile": selected.name,
        "service_url": url,
        "groups": groups,
    }

    async def run_groups() -> None:
        async with BenchmarkClientPool() as pool:
            for prompt_index, prompt_value in enumerate(prompt, start=1):
                prompt_name = f"prompt_{prompt_index}"
                for target in output_token_targets:
                    body = _request_body(
                        selected,
                        prompt=prompt_value,
                        max_tokens=target,
                        temperature=0.0,
                        top_p=1.0,
                        seed=1,
                    )
                    try:
                        observations = await _single_endpoint_observations(
                            url=url,
                            body=body,
                            runs=runs,
                            warmup_runs=warmup_runs,
                            client_pool=pool,
                            capture_raw_stream=capture_raw_stream,
                        )
                    except Exception as exc:
                        payload["failure"] = _benchmark_failure_evidence(
                            exc,
                            phase="bench-single",
                            url=url,
                            prompt_name=prompt_name,
                            output_token_target=target,
                        )
                        if output_path is not None:
                            _write_json(output_path, payload)
                        raise
                    groups.append(
                        _single_endpoint_group_payload(
                            prompt_name=prompt_name,
                            prompt=prompt_value,
                            output_token_target=target,
                            request_body=body,
                            observations=observations,
                        )
                    )
                    if output_path is not None:
                        _write_json(output_path, payload)

    asyncio.run(run_groups())
    payload["status"] = "complete"
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


def _benchmark_failure_evidence(
    exc: Exception,
    *,
    phase: str,
    url: str,
    prompt_name: str | None = None,
    output_token_target: int | None = None,
) -> dict[str, object]:
    request = getattr(exc, "request", None)
    response = getattr(exc, "response", None)
    return {
        "kind": "request_failed",
        "phase": phase,
        "url": url,
        "prompt_name": prompt_name,
        "output_token_target": output_token_target,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "request": (
            {
                "method": request.method,
                "url": str(request.url),
            }
            if isinstance(request, httpx.Request)
            else None
        ),
        "response": _benchmark_failure_response_evidence(response),
    }


def _benchmark_failure_response_evidence(response: object) -> dict[str, object] | None:
    if not isinstance(response, httpx.Response):
        return None
    return {
        "status_code": response.status_code,
        "http_version": response.http_version,
        "body": _safe_response_text(response),
    }


def _safe_response_text(response: httpx.Response) -> str | None:
    try:
        return response.text
    except Exception:
        return None


@app.command("bench-compare")
def bench_compare(
    control_json: Path = typer.Option(..., "--control-json"),
    candidate_json: Path = typer.Option(..., "--candidate-json"),
    min_meaningful_speedup: float = typer.Option(1.05, "--min-meaningful-speedup"),
    control_startup_seconds: Optional[float] = typer.Option(
        None,
        "--control-startup-seconds",
    ),
    candidate_startup_seconds: Optional[float] = typer.Option(
        None,
        "--candidate-startup-seconds",
    ),
    control_peak_gpu_memory_mib: list[float] = typer.Option(
        [],
        "--control-peak-gpu-memory-mib",
    ),
    candidate_peak_gpu_memory_mib: list[float] = typer.Option(
        [],
        "--candidate-peak-gpu-memory-mib",
    ),
    soak_passed: bool = typer.Option(False, "--soak-passed"),
    soak_seconds: Optional[float] = typer.Option(None, "--soak-seconds"),
    soak_error_count: Optional[int] = typer.Option(None, "--soak-error-count"),
    no_oom: bool = typer.Option(False, "--no-oom"),
    same_mode_mtp_parity: str = typer.Option("missing", "--same-mode-mtp-parity"),
    final_answer_quality: str = typer.Option("missing", "--final-answer-quality"),
    json_output: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Compare sequential bench-single outputs for an evidence-gated A/B."""
    if not math.isfinite(min_meaningful_speedup) or min_meaningful_speedup <= 0:
        typer.echo("--min-meaningful-speedup must be finite and positive", err=True)
        raise typer.Exit(code=2)
    control = _load_json_payload(control_json)
    candidate = _load_json_payload(candidate_json)
    payload = _compare_single_endpoint_benchmarks(
        control=control,
        candidate=candidate,
        min_meaningful_speedup=min_meaningful_speedup,
        control_startup_seconds=control_startup_seconds,
        candidate_startup_seconds=candidate_startup_seconds,
        control_peak_gpu_memory_mib=control_peak_gpu_memory_mib,
        candidate_peak_gpu_memory_mib=candidate_peak_gpu_memory_mib,
        soak_passed=soak_passed,
        soak_seconds=soak_seconds,
        soak_error_count=soak_error_count,
        no_oom=no_oom,
        same_mode_mtp_parity=same_mode_mtp_parity,
        final_answer_quality=final_answer_quality,
    )
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


def _load_json_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
        _reject_nonfinite_json_numbers(payload)
    except OSError as exc:
        raise typer.BadParameter(f"could not read JSON file: {path}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"JSON root must be an object: {path}")
    return payload


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_nonfinite_json_numbers(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_json_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json_numbers(item)


def _compare_single_endpoint_benchmarks(
    *,
    control: dict[str, object],
    candidate: dict[str, object],
    min_meaningful_speedup: float,
    control_startup_seconds: float | None,
    candidate_startup_seconds: float | None,
    control_peak_gpu_memory_mib: list[float],
    candidate_peak_gpu_memory_mib: list[float],
    soak_passed: bool,
    soak_seconds: float | None,
    soak_error_count: int | None,
    no_oom: bool,
    same_mode_mtp_parity: str,
    final_answer_quality: str,
) -> dict[str, object]:
    failure_reasons: list[str] = []
    missing_evidence: list[str] = []
    failure_reasons.extend(_p1_001_profile_scope_failures(control, candidate))
    if control.get("status") != "complete":
        failure_reasons.append("control_benchmark_incomplete")
    if candidate.get("status") != "complete":
        failure_reasons.append("candidate_benchmark_incomplete")
    if control.get("benchmark_kind") != "single_endpoint_runtime":
        failure_reasons.append("control_benchmark_kind_invalid")
    if candidate.get("benchmark_kind") != "single_endpoint_runtime":
        failure_reasons.append("candidate_benchmark_kind_invalid")

    control_groups = _single_endpoint_groups_by_key(control)
    candidate_groups = _single_endpoint_groups_by_key(candidate)
    if not control_groups:
        failure_reasons.append("control_groups_missing")
    if not candidate_groups:
        failure_reasons.append("candidate_groups_missing")
    failure_reasons.extend(
        _missing_required_target_failures(
            control_groups,
            prefix="control",
        )
    )
    failure_reasons.extend(
        _missing_required_target_failures(
            candidate_groups,
            prefix="candidate",
        )
    )

    group_reports: list[dict[str, object]] = []
    for key in sorted(set(control_groups) | set(candidate_groups)):
        control_group = control_groups.get(key)
        candidate_group = candidate_groups.get(key)
        if control_group is None:
            failure_reasons.append(f"missing_control_group:{key}")
            continue
        if candidate_group is None:
            failure_reasons.append(f"missing_candidate_group:{key}")
            continue
        report = _compare_single_endpoint_group(
            control_group=control_group,
            candidate_group=candidate_group,
            min_meaningful_speedup=min_meaningful_speedup,
        )
        group_reports.append(report)
        failure_reasons.extend(str(reason) for reason in report["failure_reasons"])
        missing_evidence.extend(str(item) for item in report["missing_evidence"])
    if control_groups and candidate_groups and not group_reports:
        failure_reasons.append("no_comparable_groups")

    _validate_positive_seconds(
        control_startup_seconds,
        missing_name="control_startup_seconds_missing",
        invalid_name="control_startup_seconds_invalid",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    _validate_positive_seconds(
        candidate_startup_seconds,
        missing_name="candidate_startup_seconds_missing",
        invalid_name="candidate_startup_seconds_invalid",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    _validate_peak_gpu_memory(
        control_peak_gpu_memory_mib,
        prefix="control",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    _validate_peak_gpu_memory(
        candidate_peak_gpu_memory_mib,
        prefix="candidate",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    _validate_soak_evidence(
        soak_passed=soak_passed,
        soak_seconds=soak_seconds,
        soak_error_count=soak_error_count,
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    if not no_oom:
        missing_evidence.append("no_oom_not_asserted")
    _validate_gate_state(
        same_mode_mtp_parity,
        failed_reason="same_mode_mtp_parity_failed",
        missing_reason="same_mode_mtp_parity_missing",
        invalid_reason="same_mode_mtp_parity_invalid",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )
    _validate_gate_state(
        final_answer_quality,
        failed_reason="final_answer_quality_failed",
        missing_reason="final_answer_quality_missing",
        invalid_reason="final_answer_quality_invalid",
        missing_evidence=missing_evidence,
        failure_reasons=failure_reasons,
    )

    if failure_reasons:
        action = "do_not_adopt"
    elif missing_evidence:
        action = "insufficient_evidence"
    else:
        action = "adopt_candidate"

    return {
        "benchmark_protocol_version": 2,
        "comparison_kind": "single_endpoint_runtime_ab",
        "control": {
            "label": control.get("label"),
            "profile": control.get("profile"),
            "startup_seconds": _finite_number_or_none(control_startup_seconds),
            "peak_gpu_memory_mib": _finite_number_list(control_peak_gpu_memory_mib),
        },
        "candidate": {
            "label": candidate.get("label"),
            "profile": candidate.get("profile"),
            "startup_seconds": _finite_number_or_none(candidate_startup_seconds),
            "peak_gpu_memory_mib": _finite_number_list(candidate_peak_gpu_memory_mib),
        },
        "min_meaningful_speedup": min_meaningful_speedup,
        "required_output_token_targets": list(P1_001_REQUIRED_OUTPUT_TOKEN_TARGETS),
        "group_comparisons": group_reports,
        "soak_passed": soak_passed,
        "soak": {
            "passed": soak_passed,
            "duration_seconds": _finite_number_or_none(soak_seconds),
            "minimum_seconds": P1_001_MIN_SOAK_SECONDS,
            "error_count": soak_error_count,
        },
        "no_oom": no_oom,
        "same_mode_mtp_parity": same_mode_mtp_parity,
        "final_answer_quality": final_answer_quality,
        "failure_reasons": _dedupe(failure_reasons),
        "missing_evidence": _dedupe(missing_evidence),
        "recommendation": {
            "action": action,
            "change_default_profile": False,
        },
    }


def _single_endpoint_groups_by_key(
    payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        prompt_name = group.get("prompt_name")
        output_token_target = group.get("output_token_target")
        prompt = group.get("prompt")
        if not isinstance(prompt_name, str) or not isinstance(prompt, str):
            continue
        if not isinstance(output_token_target, int):
            continue
        result[f"{prompt_name}|{output_token_target}|{prompt}"] = group
    return result


def _p1_001_profile_scope_failures(
    control: dict[str, object],
    candidate: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    control_profile_name = control.get("profile")
    candidate_profile_name = candidate.get("profile")
    if control_profile_name != P1_001_CONTROL_PROFILE:
        failures.append("control_profile_unexpected")
    if candidate_profile_name != P1_001_CANDIDATE_PROFILE:
        failures.append("candidate_profile_unexpected")
    try:
        profiles = _profile_set()
        control_profile = resolve_profile(P1_001_CONTROL_PROFILE, profiles)
        candidate_profile = resolve_profile(P1_001_CANDIDATE_PROFILE, profiles)
    except Exception:
        failures.append("profile_scope_unverifiable")
        return failures

    if control_profile.enforce_eager is not True:
        failures.append("control_profile_enforce_eager_not_true")
    if candidate_profile.enforce_eager is not False:
        failures.append("candidate_profile_enforce_eager_not_false")
    control_fields = asdict(control_profile)
    candidate_fields = asdict(candidate_profile)
    for field_name in ("name", "enforce_eager"):
        control_fields.pop(field_name, None)
        candidate_fields.pop(field_name, None)
    if control_fields != candidate_fields:
        failures.append("profile_settings_differ_beyond_enforce_eager")
    return failures


def _missing_required_target_failures(
    groups: dict[str, dict[str, object]],
    *,
    prefix: str,
) -> list[str]:
    available_targets = {
        group.get("output_token_target")
        for group in groups.values()
        if isinstance(group.get("output_token_target"), int)
    }
    return [
        f"{prefix}_missing_required_output_token_target:{target}"
        for target in P1_001_REQUIRED_OUTPUT_TOKEN_TARGETS
        if target not in available_targets
    ]


def _validate_positive_seconds(
    value: float | None,
    *,
    missing_name: str,
    invalid_name: str,
    missing_evidence: list[str],
    failure_reasons: list[str],
) -> None:
    if value is None:
        missing_evidence.append(missing_name)
    elif not math.isfinite(value) or value <= 0:
        failure_reasons.append(invalid_name)


def _finite_number_or_none(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _finite_number_list(values: list[float]) -> list[float | None]:
    return [_finite_number_or_none(value) for value in values]


def _sanitize_metric_summary(summary: dict[str, object]) -> dict[str, object]:
    sanitized = dict(summary)
    for key in ("median", "p10", "p90", "p95"):
        if key in sanitized:
            sanitized[key] = _finite_number_or_none(sanitized[key])
    ci = sanitized.get("bootstrap_ci_95")
    if isinstance(ci, dict):
        sanitized["bootstrap_ci_95"] = {
            "low": _finite_number_or_none(ci.get("low")),
            "high": _finite_number_or_none(ci.get("high")),
        }
    return sanitized


def _validate_peak_gpu_memory(
    values: list[float],
    *,
    prefix: str,
    missing_evidence: list[str],
    failure_reasons: list[str],
) -> None:
    if not values:
        missing_evidence.append(f"{prefix}_peak_gpu_memory_mib_missing")
        return
    if len(values) < P1_001_EXPECTED_GPU_COUNT:
        missing_evidence.append(f"{prefix}_peak_gpu_memory_mib_per_gpu_incomplete")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        failure_reasons.append(f"{prefix}_peak_gpu_memory_mib_invalid")


def _validate_soak_evidence(
    *,
    soak_passed: bool,
    soak_seconds: float | None,
    soak_error_count: int | None,
    missing_evidence: list[str],
    failure_reasons: list[str],
) -> None:
    if not soak_passed:
        missing_evidence.append("one_hour_soak_not_passed_or_not_provided")
    if soak_seconds is None:
        missing_evidence.append("soak_seconds_missing")
    elif not math.isfinite(soak_seconds) or soak_seconds <= 0:
        failure_reasons.append("soak_seconds_invalid")
    elif soak_seconds < P1_001_MIN_SOAK_SECONDS:
        failure_reasons.append("one_hour_soak_duration_insufficient")
    if soak_error_count is None:
        missing_evidence.append("soak_error_count_missing")
    elif soak_error_count < 0:
        failure_reasons.append("soak_error_count_invalid")
    elif soak_error_count > 0:
        failure_reasons.append("one_hour_soak_errors_observed")


def _validate_gate_state(
    value: str,
    *,
    failed_reason: str,
    missing_reason: str,
    invalid_reason: str,
    missing_evidence: list[str],
    failure_reasons: list[str],
) -> None:
    normalized = value.strip().lower()
    if normalized == "passed":
        return
    if normalized == "missing":
        missing_evidence.append(missing_reason)
        return
    if normalized == "failed":
        failure_reasons.append(failed_reason)
        return
    failure_reasons.append(invalid_reason)


def _compare_single_endpoint_group(
    *,
    control_group: dict[str, object],
    candidate_group: dict[str, object],
    min_meaningful_speedup: float,
) -> dict[str, object]:
    failure_reasons: list[str] = []
    missing_evidence: list[str] = []
    failure_reasons.extend(_request_body_failure_reasons(control_group, candidate_group))

    control_e2e = _group_metric_median(
        control_group,
        "e2e_output_tokens_per_second",
    )
    candidate_e2e = _group_metric_median(
        candidate_group,
        "e2e_output_tokens_per_second",
    )
    e2e_speedup = speedup(control_e2e, candidate_e2e)
    if e2e_speedup is None or e2e_speedup < min_meaningful_speedup:
        failure_reasons.append("meaningful_e2e_speedup_missing")

    control_ttft = _group_metric_summary(control_group, "ttft_ms")
    candidate_ttft = _group_metric_summary(candidate_group, "ttft_ms")
    if not _metric_summary_has_median(control_ttft) or not _metric_summary_has_median(
        candidate_ttft
    ):
        missing_evidence.append("ttft_evidence_missing")

    control_tpot = _group_metric_summary(control_group, "tpot_ms")
    candidate_tpot = _group_metric_summary(candidate_group, "tpot_ms")
    if not _metric_summary_has_median(control_tpot) or not _metric_summary_has_median(
        candidate_tpot
    ):
        missing_evidence.append("tpot_evidence_missing")

    acceptance_rate = _non_inferiority_report(
        control_group,
        candidate_group,
        metric_name="acceptance_rate_delta",
        margin=P1_001_ACCEPTANCE_RATE_MARGIN,
    )
    if acceptance_rate["status"] == "missing":
        missing_evidence.append("mtp_acceptance_evidence_missing")
    elif acceptance_rate["status"] == "failed":
        failure_reasons.append("mtp_acceptance_regression")

    acceptance_length = _non_inferiority_report(
        control_group,
        candidate_group,
        metric_name="mean_acceptance_length_delta",
        margin=P1_001_MEAN_ACCEPTANCE_LENGTH_MARGIN,
    )
    if acceptance_length["status"] == "missing":
        missing_evidence.append("mtp_mean_acceptance_length_evidence_missing")
    elif acceptance_length["status"] == "failed":
        failure_reasons.append("mtp_mean_acceptance_length_regression")

    control_repeatability = _group_within_backend_repeatability(control_group)
    candidate_repeatability = _group_within_backend_repeatability(candidate_group)
    if control_repeatability["status"] == "missing":
        missing_evidence.append("control_raw_token_evidence_missing")
    elif control_repeatability["status"] == "failed":
        failure_reasons.append("control_within_backend_repeatability_failed")
    if candidate_repeatability["status"] == "missing":
        missing_evidence.append("candidate_raw_token_evidence_missing")
    elif candidate_repeatability["status"] == "failed":
        failure_reasons.append("candidate_within_backend_repeatability_failed")

    cross_parity = _group_token_parity(control_group, candidate_group)

    return {
        "prompt_name": control_group.get("prompt_name"),
        "output_token_target": control_group.get("output_token_target"),
        "control": {
            "e2e_output_tokens_per_second_median": control_e2e,
            "ttft_ms": control_ttft,
            "tpot_ms": control_tpot,
            "mtp_acceptance_rate": acceptance_rate["control"],
            "mtp_mean_acceptance_length": acceptance_length["control"],
            "within_backend_repeatability": control_repeatability,
        },
        "candidate": {
            "e2e_output_tokens_per_second_median": candidate_e2e,
            "ttft_ms": candidate_ttft,
            "tpot_ms": candidate_tpot,
            "mtp_acceptance_rate": acceptance_rate["candidate"],
            "mtp_mean_acceptance_length": acceptance_length["candidate"],
            "within_backend_repeatability": candidate_repeatability,
        },
        "e2e_speedup": e2e_speedup,
        "within_backend_repeatability": {
            "control": control_repeatability,
            "candidate": candidate_repeatability,
        },
        "same_execution_mode_mtp_parity": {
            "status": "not_measured",
            "reason": "requires no-MTP baseline for the same execution mode",
        },
        "cross_execution_mode_parity": cross_parity,
        "deterministic_parity": cross_parity["deterministic_parity"],
        "parity_reason": cross_parity["reason"],
        "acceptance_non_inferiority": acceptance_rate,
        "mean_acceptance_length_non_inferiority": acceptance_length,
        "failure_reasons": _dedupe(failure_reasons),
        "missing_evidence": _dedupe(missing_evidence),
    }


def _request_body_failure_reasons(
    control_group: dict[str, object],
    candidate_group: dict[str, object],
) -> list[str]:
    control_body = control_group.get("request_body")
    candidate_body = candidate_group.get("request_body")
    if not isinstance(control_body, dict) or not isinstance(candidate_body, dict):
        return ["request_body_missing"]
    if control_body != candidate_body:
        return ["request_body_mismatch"]
    return []


def _non_inferiority_report(
    control_group: dict[str, object],
    candidate_group: dict[str, object],
    *,
    metric_name: str,
    margin: float,
) -> dict[str, object]:
    control_values = _group_mtp_delta_values(control_group, metric_name)
    candidate_values = _group_mtp_delta_values(candidate_group, metric_name)
    control_clean = _finite_values(control_values)
    candidate_clean = _finite_values(candidate_values)
    control_summary = metric_summary(control_clean)
    candidate_summary = metric_summary(candidate_clean)
    paired_diffs = [
        candidate - control
        for control, candidate in zip(control_values, candidate_values)
        if isinstance(control, (int, float))
        and isinstance(candidate, (int, float))
        and math.isfinite(float(control))
        and math.isfinite(float(candidate))
    ]
    diff_summary = metric_summary(paired_diffs)
    ci = diff_summary.get("bootstrap_ci_95")
    ci_low = ci.get("low") if isinstance(ci, dict) else None
    if not paired_diffs or not isinstance(ci_low, (int, float)):
        status = "missing"
    elif float(ci_low) < margin:
        status = "failed"
    else:
        status = "passed"
    return {
        "metric": metric_name,
        "status": status,
        "margin": margin,
        "control": control_summary,
        "candidate": candidate_summary,
        "candidate_minus_control": diff_summary,
    }


def _finite_values(values: list[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _group_within_backend_repeatability(group: dict[str, object]) -> dict[str, object]:
    observations = _group_observations(group)
    sequences: list[list[int]] = []
    for observation in observations:
        if not observation.get("parity_ready"):
            return {
                "status": "missing",
                "reason": "raw_token_evidence_unavailable",
                "observation_count": len(observations),
            }
        tokens = observation.get("output_token_ids")
        if not _is_token_id_sequence(tokens):
            return {
                "status": "missing",
                "reason": "raw_token_ids_missing",
                "observation_count": len(observations),
            }
        sequences.append(list(tokens))
    if not sequences:
        return {"status": "missing", "reason": "observations_missing"}
    unique = {_token_sequence_hash(sequence) for sequence in sequences}
    return {
        "status": "passed" if len(unique) == 1 else "failed",
        "observation_count": len(sequences),
        "unique_sequence_count": len(unique),
        "unique_sequence_hashes": sorted(unique),
        "sequence_lengths": [len(sequence) for sequence in sequences],
    }


def _is_token_id_sequence(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _token_sequence_hash(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _token_sequence_diagnostics(
    control_tokens: list[int],
    candidate_tokens: list[int],
) -> dict[str, object]:
    max_len = max(len(control_tokens), len(candidate_tokens))
    lcp = 0
    for left, right in zip(control_tokens, candidate_tokens):
        if left != right:
            break
        lcp += 1
    same_position = sum(
        1 for left, right in zip(control_tokens, candidate_tokens) if left == right
    )
    exact = control_tokens == candidate_tokens
    return {
        "exact_match": exact,
        "control_length": len(control_tokens),
        "candidate_length": len(candidate_tokens),
        "output_length_equal": len(control_tokens) == len(candidate_tokens),
        "longest_common_prefix_tokens": lcp,
        "first_divergence_position_0_based": None if exact else lcp,
        "matching_token_percentage_same_position_over_max_len": (
            same_position / max_len if max_len else 1.0
        ),
    }


def _group_metric_median(group: dict[str, object], metric_name: str) -> float | None:
    summary = _group_metric_summary(group, metric_name)
    median = summary.get("median")
    if not isinstance(median, (int, float)):
        return None
    median_float = float(median)
    return median_float if math.isfinite(median_float) else None


def _metric_summary_has_median(summary: dict[str, object]) -> bool:
    median = summary.get("median")
    return isinstance(median, (int, float)) and math.isfinite(float(median))


def _group_metric_summary(
    group: dict[str, object],
    metric_name: str,
) -> dict[str, object]:
    statistics_payload = group.get("statistics")
    if isinstance(statistics_payload, dict):
        summary = statistics_payload.get(metric_name)
        if isinstance(summary, dict):
            return _sanitize_metric_summary(summary)
    return metric_summary(
        [
            _nested_float(observation, "result", metric_name)
            for observation in _group_observations(group)
        ]
    )


def _group_mtp_delta_values(
    group: dict[str, object],
    metric_name: str,
) -> list[float | None]:
    return [
        _nested_float(observation, "mtp_metrics_delta", metric_name)
        for observation in _group_observations(group)
    ]


def _group_token_parity(
    control_group: dict[str, object],
    candidate_group: dict[str, object],
) -> dict[str, object]:
    control_observations = _group_observations(control_group)
    candidate_observations = _group_observations(candidate_group)
    if len(control_observations) != len(candidate_observations):
        return {"deterministic_parity": False, "reason": "observation_count_mismatch"}
    diagnostics: list[dict[str, object]] = []
    for control_observation, candidate_observation in zip(
        control_observations,
        candidate_observations,
    ):
        if not control_observation.get("parity_ready") or not candidate_observation.get(
            "parity_ready"
        ):
            return {"deterministic_parity": False, "reason": "token_parity_unavailable"}
        control_tokens = control_observation.get("output_token_ids")
        candidate_tokens = candidate_observation.get("output_token_ids")
        if not _is_token_id_sequence(control_tokens) or not _is_token_id_sequence(
            candidate_tokens
        ):
            return {"deterministic_parity": False, "reason": "token_parity_unavailable"}
        diagnostic = _token_sequence_diagnostics(
            list(control_tokens),
            list(candidate_tokens),
        )
        diagnostic["index"] = control_observation.get("index")
        diagnostics.append(diagnostic)
        if not diagnostic["exact_match"]:
            return {
                "deterministic_parity": False,
                "reason": "deterministic_parity_failed",
                "diagnostics": diagnostics,
            }
    return {
        "deterministic_parity": True,
        "reason": "token_ids_match",
        "diagnostics": diagnostics,
    }


def _group_observations(group: dict[str, object]) -> list[dict[str, object]]:
    observations = group.get("observations")
    if not isinstance(observations, list):
        return []
    return [
        observation
        for observation in observations
        if isinstance(observation, dict)
    ]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _single_endpoint_group_payload(
    *,
    prompt_name: str,
    prompt: str,
    output_token_target: int,
    request_body: dict,
    observations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "output_token_target": output_token_target,
        "request_body": request_body,
        "observations": observations,
        "statistics": {
            "e2e_output_tokens_per_second": metric_summary(
                [
                    _nested_float(
                        observation,
                        "result",
                        "e2e_output_tokens_per_second",
                    )
                    for observation in observations
                ]
            ),
            "ttft_ms": metric_summary(
                [
                    _nested_float(observation, "result", "ttft_ms")
                    for observation in observations
                ]
            ),
            "generated_ttft_ms": metric_summary(
                [
                    _nested_float(observation, "result", "generated_ttft_ms")
                    for observation in observations
                ]
            ),
            "visible_content_ttft_ms": metric_summary(
                [
                    _nested_float(observation, "result", "visible_content_ttft_ms")
                    for observation in observations
                ]
            ),
            "tpot_ms": metric_summary(
                [
                    _nested_float(observation, "result", "tpot_ms")
                    for observation in observations
                ]
            ),
        },
    }


async def _single_endpoint_observations(
    *,
    url: str,
    body: dict,
    runs: int,
    warmup_runs: int,
    client_pool: BenchmarkClientPool | None = None,
    capture_raw_stream: bool = False,
) -> list[dict[str, object]]:
    if client_pool is None:
        async with BenchmarkClientPool() as pool:
            return await _single_endpoint_observations(
                url=url,
                body=body,
                runs=runs,
                warmup_runs=warmup_runs,
                client_pool=pool,
                capture_raw_stream=capture_raw_stream,
            )

    client = client_pool.client_for(url)
    for _ in range(warmup_runs):
        await _measure(url, body, client=client)

    observations: list[dict[str, object]] = []
    for index in range(1, runs + 1):
        metrics_before = await _fetch_mtp_metrics(url, client=client)
        text, result = await _measure(
            url,
            body,
            client=client,
            capture_raw_stream=capture_raw_stream,
        )
        output_token_ids = result.raw_output_token_ids
        tokenization_status = (
            result.token_count_validation_status
            if output_token_ids is not None
            else "raw_unavailable"
        )
        metrics_after = await _fetch_mtp_metrics(url, client=client)
        observations.append(
            {
                "index": index,
                "result": result.__dict__,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_token_ids": output_token_ids,
                "raw_output_token_ids": output_token_ids,
                "reasoning_text": result.reasoning_text,
                "visible_content": result.visible_content,
                "tokenization_status": tokenization_status,
                "timing_evidence_valid": result.timing_evidence_valid,
                "parity_ready": (
                    output_token_ids is not None and result.timing_evidence_valid
                ),
                "mtp_metrics_before": metrics_before,
                "mtp_metrics_after": metrics_after,
                "mtp_metrics_delta": mtp_metric_delta(metrics_before, metrics_after),
            }
        )
    return observations


def _nested_float(payload: dict[str, object], *keys: str) -> float | None:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if not isinstance(value, (int, float)):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _write_benchmark_artifacts(
    *,
    artifact_root: Path,
    artifact_id: str | None,
    profile: ModelProfile,
    summary_payload: dict,
    request_body: dict,
    mtp_url: str,
    baseline_url: str,
    metrics_before_text: str,
    metrics_after_text: str,
    runtime_manifest_path: Path | None,
) -> Path:
    artifact_name = artifact_id or _default_artifact_id(profile.name)
    artifact_dir = artifact_root / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    before_metrics = parse_mtp_metrics(metrics_before_text)
    after_metrics = parse_mtp_metrics(metrics_after_text)
    metrics_delta = mtp_metric_delta(before_metrics, after_metrics)
    manifest = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "git_sha": _git_sha(),
        "package_version": __version__,
        "profile": profile.name,
        "target_model": profile.target,
        "served_model_name": DEFAULT_MODEL_ALIAS,
        "drafter_model": profile.drafter,
        "quantization": profile.quantization,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "max_model_len": profile.max_model_len,
        "num_speculative_tokens": profile.num_speculative_tokens,
        "service_urls": {"mtp": mtp_url, "baseline": baseline_url},
        "output_token_targets": [summary_payload["output_token_target"]],
        "prompt_names": [summary_payload["prompt_name"]],
        "runtime_manifest_source": (
            "provided"
            if runtime_manifest_path is not None and runtime_manifest_path.exists()
            else "unavailable"
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    _write_json(artifact_dir / "results.json", summary_payload)
    _write_json(artifact_dir / "metrics-delta.json", metrics_delta)
    _write_json(
        artifact_dir / "runtime-manifest.json",
        _load_runtime_manifest(runtime_manifest_path),
    )
    _write_json(artifact_dir / "request-payloads.json", {"chat_completion": request_body})
    (artifact_dir / "results.md").write_text(
        _render_results_markdown(summary_payload),
        encoding="utf-8",
    )
    (artifact_dir / "metrics-before.prom").write_text(metrics_before_text, encoding="utf-8")
    (artifact_dir / "metrics-after.prom").write_text(metrics_after_text, encoding="utf-8")
    (artifact_dir / "environment.txt").write_text(_environment_text(), encoding="utf-8")
    (artifact_dir / "nvidia-smi.csv").write_text(_nvidia_smi_csv(), encoding="utf-8")
    (artifact_dir / "README.md").write_text(
        "# Benchmark Artifact\n\n"
        "This directory was generated by `vllm-mtp bench` using benchmark "
        f"protocol version {BENCHMARK_PROTOCOL_VERSION}.\n",
        encoding="utf-8",
    )
    return artifact_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_runtime_manifest(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_artifact_id(profile_name: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{_safe_artifact_component(socket.gethostname())}-{_safe_artifact_component(profile_name)}"


def _safe_artifact_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return cleaned.strip("-") or "unknown"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _environment_text() -> str:
    return "\n".join(
        [
            f"python={sys.version.split()[0]}",
            f"platform={platform.platform()}",
            f"hostname={socket.gethostname()}",
            f"package_version={__version__}",
        ]
    ) + "\n"


def _nvidia_smi_csv() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return "unavailable\n"


def _render_results_markdown(summary_payload: dict) -> str:
    stats = summary_payload.get("statistics") or {}
    baseline = (
        ((stats.get("baseline") or {}).get("e2e_output_tokens_per_second") or {})
        .get("median")
    )
    mtp = ((stats.get("mtp") or {}).get("e2e_output_tokens_per_second") or {}).get(
        "median"
    )
    speedup_value = (stats.get("speedup") or {}).get("median")
    return (
        "# Benchmark Results\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"| Baseline median e2e output tok/s | {_format_optional_float(baseline)} |\n"
        f"| MTP median e2e output tok/s | {_format_optional_float(mtp)} |\n"
        f"| Median speedup | {_format_optional_float(speedup_value)} |\n"
    )


def _format_optional_float(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else "unknown"


@app.command("bench-matrix")
def bench_matrix(
    profile: str = typer.Option("safe80", "--profile"),
    mtp_url: Optional[str] = typer.Option(None, "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    prompt: list[str] = typer.Option([], "--prompt"),
    num_speculative_tokens: list[int] = typer.Option(
        [], "--num-speculative-tokens"
    ),
    output_token_target: list[int] = typer.Option([], "--output-token-target"),
    depth_mtp_url: list[str] = typer.Option(
        [],
        "--depth-mtp-url",
        help="Depth-specific MTP endpoint as N=URL; required for multi-depth sweeps.",
    ),
    runs: int = typer.Option(10, "--runs"),
    warmup_runs: int = typer.Option(2, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
    capture_raw_stream: bool = typer.Option(False, "--capture-raw-stream"),
) -> None:
    """Sweep MTP vs baseline across prompts x num_speculative_tokens."""
    if not prompt:
        typer.echo("at least one --prompt required", err=True)
        raise typer.Exit(code=2)
    if not num_speculative_tokens:
        typer.echo("at least one --num-speculative-tokens required", err=True)
        raise typer.Exit(code=2)
    if any(value <= 0 for value in num_speculative_tokens):
        typer.echo("--num-speculative-tokens must be positive", err=True)
        raise typer.Exit(code=2)
    mtp_urls = _parse_depth_mtp_urls(depth_mtp_url)
    if mtp_urls:
        missing = [n for n in num_speculative_tokens if n not in mtp_urls]
        if missing:
            typer.echo(
                "missing --depth-mtp-url for num_speculative_tokens: "
                + ", ".join(str(n) for n in missing),
                err=True,
            )
            raise typer.Exit(code=2)
    elif len(set(num_speculative_tokens)) > 1:
        typer.echo(
            "multi-depth bench-matrix requires --depth-mtp-url N=URL for each "
            "--num-speculative-tokens; a single --mtp-url cannot change live "
            "vLLM speculative depth",
            err=True,
        )
        raise typer.Exit(code=2)
    elif mtp_url is None:
        typer.echo("--mtp-url is required unless --depth-mtp-url is provided", err=True)
        raise typer.Exit(code=2)

    selected_base = resolve_profile(profile, _profile_set())
    output_token_targets = output_token_target or [64]

    async def run_matrix() -> list[dict]:
        results: list[dict] = []
        async with BenchmarkClientPool() as pool:
            # Use enumerate() rather than prompt.index(prompt_value) so duplicate
            # prompts get distinct prompt_name labels.
            for prompt_index, prompt_value in enumerate(prompt, start=1):
                for n in num_speculative_tokens:
                    for target in output_token_targets:
                        # dataclasses.replace() is the canonical copy-with-override for
                        # frozen dataclasses; avoids touching the private __dict__.
                        adjusted = replace(selected_base, num_speculative_tokens=n)
                        selected_mtp_url = mtp_urls.get(n) or mtp_url
                        if selected_mtp_url is None:
                            typer.echo("missing MTP URL", err=True)
                            raise typer.Exit(code=2)
                        observations = await _single_bench(
                            profile=adjusted,
                            prompt=prompt_value,
                            max_tokens=target,
                            mtp_url=selected_mtp_url,
                            baseline_url=baseline_url,
                            runs=runs,
                            warmup_runs=warmup_runs,
                            client_pool=pool,
                            capture_raw_stream=capture_raw_stream,
                        )
                        summary = BenchmarkSummary(
                            profile=adjusted.name,
                            prompt_name=f"prompt_{prompt_index}",
                            prompt=prompt_value,
                            output_token_target=target,
                            num_speculative_tokens=n,
                            observations=observations,
                        )
                        results.append(summary.to_dict())
        return results

    results = asyncio.run(run_matrix())
    rendered = json.dumps(results, indent=2)
    if json_output:
        Path(json_output).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


def _parse_depth_mtp_urls(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            typer.echo("--depth-mtp-url must use N=URL", err=True)
            raise typer.Exit(code=2)
        raw_depth, url = value.split("=", 1)
        try:
            depth = int(raw_depth)
        except ValueError:
            typer.echo("--depth-mtp-url depth must be an integer", err=True)
            raise typer.Exit(code=2)
        if depth <= 0 or not url:
            typer.echo("--depth-mtp-url must use positive N=URL", err=True)
            raise typer.Exit(code=2)
        if url in result.values():
            typer.echo(
                "each speculative depth requires a distinct --depth-mtp-url endpoint",
                err=True,
            )
            raise typer.Exit(code=2)
        result[depth] = url
    return result
