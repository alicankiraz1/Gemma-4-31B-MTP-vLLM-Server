from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

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
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _http_client(base_url: str) -> httpx.AsyncClient:
    transport = _build_transport()
    if transport is not None:
        return httpx.AsyncClient(
            transport=transport,
            base_url=base_url,
            timeout=_vllm_http_timeout(),
        )
    return httpx.AsyncClient(base_url=base_url, timeout=_vllm_http_timeout())


def _vllm_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)


async def _measure(
    base_url: str,
    body: dict,
) -> tuple[str, BenchmarkEndpointResult]:
    async with _http_client(base_url) as http:
        start = time.perf_counter()
        token_times: list[float] = []
        output_parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        async with http.stream("POST", "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ").strip()
                if raw == "[DONE]":
                    break
                payload = json.loads(raw)
                usage = payload.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        token_times.append(time.perf_counter())
                        output_parts.append(str(content))
        end = time.perf_counter()
    text = "".join(output_parts)
    if completion_tokens is None and text:
        completion_tokens = await _count_output_tokens(base_url, text)
    observed_completion_tokens = completion_tokens
    total_latency_ms = (end - start) * 1000.0
    ttft_ms = (token_times[0] - start) * 1000.0 if token_times else None
    token_timing_complete = (
        observed_completion_tokens is not None
        and observed_completion_tokens > 0
        and len(token_times) == observed_completion_tokens
    )
    intervals_ms = (
        [
            (right - left) * 1000.0
            for left, right in zip(token_times, token_times[1:])
        ]
        if token_timing_complete
        else []
    )
    elapsed_seconds = max(end - start, 0.0)
    throughput = (
        observed_completion_tokens / elapsed_seconds
        if elapsed_seconds > 0
        and observed_completion_tokens is not None
        and observed_completion_tokens > 0
        else None
    )
    result = BenchmarkEndpointResult(
        e2e_output_tokens_per_second=throughput,
        ttft_ms=ttft_ms,
        tpot_ms=(
            ((token_times[-1] - token_times[0]) * 1000.0)
            / (observed_completion_tokens - 1)
            if token_timing_complete
            and ttft_ms is not None
            and observed_completion_tokens is not None
            and observed_completion_tokens > 1
            else None
        ),
        inter_token_latency_ms_p50=percentile(intervals_ms, 50),
        inter_token_latency_ms_p95=percentile(intervals_ms, 95),
        total_latency_ms=total_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=observed_completion_tokens,
    )
    return text, result


async def _fetch_mtp_metrics(base_url: str) -> dict:
    return parse_mtp_metrics(await _fetch_metrics_text(base_url))


async def _fetch_metrics_text(base_url: str) -> str:
    try:
        async with _http_client(base_url) as http:
            response = await http.get("/metrics")
            response.raise_for_status()
            return response.text
    except Exception:
        return ""


async def _tokenize_visible_text(base_url: str, text: str) -> dict | None:
    try:
        async with _http_client(base_url) as http:
            response = await http.post(
                "/tokenize",
                json={
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


async def _count_output_tokens(base_url: str, text: str) -> int | None:
    return _count_from_tokenize_payload(await _tokenize_visible_text(base_url, text))


async def _tokenize_output(base_url: str, text: str) -> list[int] | None:
    return _tokens_from_tokenize_payload(await _tokenize_visible_text(base_url, text))


async def _deterministic_parity_for_outputs(
    *,
    baseline_url: str,
    mtp_url: str,
    baseline_text: str,
    mtp_text: str,
) -> tuple[bool | None, str]:
    baseline_tokens = await _tokenize_output(baseline_url, baseline_text)
    mtp_tokens = await _tokenize_output(mtp_url, mtp_text)
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
) -> list[BenchmarkObservation]:
    body = _request_body(
        profile,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=1,
    )

    for _ in range(warmup_runs):
        await _measure(mtp_url, body)
        await _measure(baseline_url, body)

    observations: list[BenchmarkObservation] = []
    for idx in range(1, runs + 1):
        if idx % 2:
            mtp_metrics_before = await _fetch_mtp_metrics(mtp_url)
            mtp_text, mtp_result = await _measure(mtp_url, body)
            mtp_metrics_after = await _fetch_mtp_metrics(mtp_url)
            no_text, baseline_result = await _measure(baseline_url, body)
        else:
            no_text, baseline_result = await _measure(baseline_url, body)
            mtp_metrics_before = await _fetch_mtp_metrics(mtp_url)
            mtp_text, mtp_result = await _measure(mtp_url, body)
            mtp_metrics_after = await _fetch_mtp_metrics(mtp_url)
        parity, parity_basis = await _deterministic_parity_for_outputs(
            baseline_url=baseline_url,
            mtp_url=mtp_url,
            baseline_text=no_text,
            mtp_text=mtp_text,
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
) -> None:
    """Compare MTP vs baseline vLLM endpoints for a single prompt."""
    selected = resolve_profile(profile, _profile_set())
    metrics_before_text = (
        asyncio.run(_fetch_metrics_text(mtp_url)) if artifact_root is not None else ""
    )
    observations = asyncio.run(
        _single_bench(
            profile=selected,
            prompt=prompt,
            max_tokens=max_tokens,
            mtp_url=mtp_url,
            baseline_url=baseline_url,
            runs=runs,
            warmup_runs=warmup_runs,
        )
    )
    metrics_after_text = (
        asyncio.run(_fetch_metrics_text(mtp_url)) if artifact_root is not None else ""
    )
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
        "benchmark_protocol_version": 2,
        "benchmark_kind": "single_endpoint_runtime",
        "status": "in_progress",
        "label": label,
        "profile": selected.name,
        "service_url": url,
        "groups": groups,
    }
    for prompt_index, prompt_value in enumerate(prompt, start=1):
        for target in output_token_targets:
            body = _request_body(
                selected,
                prompt=prompt_value,
                max_tokens=target,
                temperature=0.0,
                top_p=1.0,
                seed=1,
            )
            observations = asyncio.run(
                _single_endpoint_observations(
                    url=url,
                    body=body,
                    runs=runs,
                    warmup_runs=warmup_runs,
                )
            )
            groups.append(
                _single_endpoint_group_payload(
                    prompt_name=f"prompt_{prompt_index}",
                    prompt=prompt_value,
                    output_token_target=target,
                    request_body=body,
                    observations=observations,
                )
            )
            if output_path is not None:
                _write_json(output_path, payload)
    payload["status"] = "complete"
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


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


def _compare_single_endpoint_group(
    *,
    control_group: dict[str, object],
    candidate_group: dict[str, object],
    min_meaningful_speedup: float,
) -> dict[str, object]:
    failure_reasons: list[str] = []
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
        failure_reasons.append("ttft_evidence_missing")

    control_tpot = _group_metric_summary(control_group, "tpot_ms")
    candidate_tpot = _group_metric_summary(candidate_group, "tpot_ms")
    if not _metric_summary_has_median(control_tpot) or not _metric_summary_has_median(
        candidate_tpot
    ):
        failure_reasons.append("tpot_evidence_missing")

    control_acceptance_rate = metric_summary(
        _group_mtp_delta_values(control_group, "acceptance_rate_delta")
    )
    candidate_acceptance_rate = metric_summary(
        _group_mtp_delta_values(candidate_group, "acceptance_rate_delta")
    )
    control_acceptance_rate_median = control_acceptance_rate["median"]
    candidate_acceptance_rate_median = candidate_acceptance_rate["median"]
    if not isinstance(control_acceptance_rate_median, (int, float)) or not isinstance(
        candidate_acceptance_rate_median,
        (int, float),
    ):
        failure_reasons.append("mtp_acceptance_evidence_missing")
    elif candidate_acceptance_rate_median < control_acceptance_rate_median:
        failure_reasons.append("mtp_acceptance_regression")

    control_acceptance_length = metric_summary(
        _group_mtp_delta_values(control_group, "mean_acceptance_length_delta")
    )
    candidate_acceptance_length = metric_summary(
        _group_mtp_delta_values(candidate_group, "mean_acceptance_length_delta")
    )
    control_acceptance_length_median = control_acceptance_length["median"]
    candidate_acceptance_length_median = candidate_acceptance_length["median"]
    if not isinstance(
        control_acceptance_length_median,
        (int, float),
    ) or not isinstance(candidate_acceptance_length_median, (int, float)):
        failure_reasons.append("mtp_mean_acceptance_length_evidence_missing")
    elif candidate_acceptance_length_median < control_acceptance_length_median:
        failure_reasons.append("mtp_mean_acceptance_length_regression")

    parity = _group_token_parity(control_group, candidate_group)
    if not parity["deterministic_parity"]:
        failure_reasons.append(str(parity["reason"]))

    return {
        "prompt_name": control_group.get("prompt_name"),
        "output_token_target": control_group.get("output_token_target"),
        "control": {
            "e2e_output_tokens_per_second_median": control_e2e,
            "ttft_ms": control_ttft,
            "tpot_ms": control_tpot,
            "mtp_acceptance_rate": control_acceptance_rate,
            "mtp_mean_acceptance_length": control_acceptance_length,
        },
        "candidate": {
            "e2e_output_tokens_per_second_median": candidate_e2e,
            "ttft_ms": candidate_ttft,
            "tpot_ms": candidate_tpot,
            "mtp_acceptance_rate": candidate_acceptance_rate,
            "mtp_mean_acceptance_length": candidate_acceptance_length,
        },
        "e2e_speedup": e2e_speedup,
        "deterministic_parity": parity["deterministic_parity"],
        "parity_reason": parity["reason"],
        "failure_reasons": _dedupe(failure_reasons),
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
    for control_observation, candidate_observation in zip(
        control_observations,
        candidate_observations,
    ):
        if not control_observation.get("parity_ready") or not candidate_observation.get(
            "parity_ready"
        ):
            return {"deterministic_parity": False, "reason": "token_parity_unavailable"}
        if control_observation.get("output_token_ids") != candidate_observation.get(
            "output_token_ids"
        ):
            return {
                "deterministic_parity": False,
                "reason": "deterministic_parity_failed",
            }
    return {"deterministic_parity": True, "reason": "token_ids_match"}


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
) -> list[dict[str, object]]:
    for _ in range(warmup_runs):
        await _measure(url, body)

    observations: list[dict[str, object]] = []
    for index in range(1, runs + 1):
        metrics_before = await _fetch_mtp_metrics(url)
        text, result = await _measure(url, body)
        output_token_ids = await _tokenize_output(url, text)
        tokenization_status = (
            "available" if output_token_ids is not None else "unavailable"
        )
        metrics_after = await _fetch_mtp_metrics(url)
        observations.append(
            {
                "index": index,
                "result": result.__dict__,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_token_ids": output_token_ids,
                "tokenization_status": tokenization_status,
                "parity_ready": output_token_ids is not None,
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
        "benchmark_protocol_version": 2,
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
        "protocol version 2.\n",
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
    results: list[dict] = []
    output_token_targets = output_token_target or [64]
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
                observations = asyncio.run(
                    _single_bench(
                        profile=adjusted,
                        prompt=prompt_value,
                        max_tokens=target,
                        mtp_url=selected_mtp_url,
                        baseline_url=baseline_url,
                        runs=runs,
                        warmup_runs=warmup_runs,
                    )
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
