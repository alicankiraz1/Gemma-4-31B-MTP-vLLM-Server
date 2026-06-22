from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import httpx
import typer
import uvicorn

from gemma4_mtp_vllm import __version__
from gemma4_mtp_vllm.benchmarking import (
    BenchmarkObservation,
    BenchmarkSummary,
    deterministic_parity,
    median_optional,
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
) -> dict:
    return {
        "model": DEFAULT_MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


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
) -> tuple[str, float]:
    async with _http_client(base_url) as http:
        start = time.perf_counter()
        response = await http.post("/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    completion_tokens = (payload.get("usage") or {}).get("completion_tokens") or 1
    # Test seam: handlers can inject a deterministic TPS value to keep
    # in-process MockTransport timing-independent.
    test_tps = payload.get("vllm_tps_for_test")
    if isinstance(test_tps, (int, float)):
        return text, float(test_tps)
    tps = completion_tokens / elapsed if elapsed > 0 else None
    return text, float(tps) if tps else 0.0


async def _fetch_mtp_metrics(base_url: str) -> dict:
    try:
        async with _http_client(base_url) as http:
            response = await http.get("/metrics")
            response.raise_for_status()
            return parse_mtp_metrics(response.text)
    except Exception:
        return parse_mtp_metrics("")


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
    )

    for _ in range(warmup_runs):
        await _measure(mtp_url, body)
        await _measure(baseline_url, body)

    observations: list[BenchmarkObservation] = []
    for idx in range(1, runs + 1):
        mtp_metrics_before = await _fetch_mtp_metrics(mtp_url)
        mtp_text, mtp_tps = await _measure(mtp_url, body)
        mtp_metrics_after = await _fetch_mtp_metrics(mtp_url)
        no_text, no_tps = await _measure(baseline_url, body)
        observations.append(
            BenchmarkObservation(
                index=idx,
                no_draft_generation_tps=no_tps,
                mtp_generation_tps=mtp_tps,
                speedup=speedup(no_tps, mtp_tps),
                deterministic_parity=deterministic_parity(
                    no_text, mtp_text, temperature=0.0, top_p=1.0
                ),
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
    max_tokens: int = typer.Option(64, "--max-tokens"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    """Compare MTP vs baseline vLLM endpoints for a single prompt."""
    selected = resolve_profile(profile, _profile_set())
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
    summary = BenchmarkSummary(
        profile=selected.name,
        prompt_name="default",
        prompt=prompt,
        num_speculative_tokens=selected.num_speculative_tokens,
        observations=observations,
        median_no_draft_generation_tps=median_optional(
            [obs.no_draft_generation_tps for obs in observations]
        ),
        median_mtp_generation_tps=median_optional(
            [obs.mtp_generation_tps for obs in observations]
        ),
        median_speedup=median_optional([obs.speedup for obs in observations]),
    )
    payload = summary.to_dict()
    rendered = json.dumps(payload, indent=2)
    if json_output:
        Path(json_output).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@app.command("bench-matrix")
def bench_matrix(
    profile: str = typer.Option("safe80", "--profile"),
    mtp_url: Optional[str] = typer.Option(None, "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    prompt: list[str] = typer.Option([], "--prompt"),
    num_speculative_tokens: list[int] = typer.Option(
        [], "--num-speculative-tokens"
    ),
    depth_mtp_url: list[str] = typer.Option(
        [],
        "--depth-mtp-url",
        help="Depth-specific MTP endpoint as N=URL; required for multi-depth sweeps.",
    ),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
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
    # Use enumerate() rather than prompt.index(prompt_value) so duplicate
    # prompts get distinct prompt_name labels.
    for prompt_index, prompt_value in enumerate(prompt, start=1):
        for n in num_speculative_tokens:
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
                    max_tokens=64,
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
                num_speculative_tokens=n,
                observations=observations,
                median_no_draft_generation_tps=median_optional(
                    [obs.no_draft_generation_tps for obs in observations]
                ),
                median_mtp_generation_tps=median_optional(
                    [obs.mtp_generation_tps for obs in observations]
                ),
                median_speedup=median_optional([obs.speedup for obs in observations]),
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
        result[depth] = url
    return result
