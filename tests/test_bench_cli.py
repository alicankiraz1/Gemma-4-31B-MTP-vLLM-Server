from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from gemma4_mtp_vllm import cli as cli_module
from gemma4_mtp_vllm.cli import app


runner = CliRunner()


class _FakeBenchClock:
    def __init__(self) -> None:
        self.now = 0.0
        self._start = 0.0
        self._scheduled: list[float] = []

    def perf_counter(self) -> float:
        if self._scheduled:
            value = self._scheduled.pop(0)
            if not self._scheduled:
                self.now = value + 1.0
            return value
        self._start = self.now
        return self.now

    def schedule(self, *, completion_tokens: int, tps: float) -> None:
        elapsed = completion_tokens / tps
        self._scheduled = [
            self._start + (elapsed * (index + 1) / completion_tokens)
            for index in range(completion_tokens)
        ]
        self._scheduled.append(self._start + elapsed)


def _install_fake_clock(monkeypatch) -> _FakeBenchClock:
    clock = _FakeBenchClock()
    monkeypatch.setattr("gemma4_mtp_vllm.cli.time.perf_counter", clock.perf_counter)
    return clock


def _stream_response(*, text: str) -> httpx.Response:
    chunks = "".join(
        f'data: {{"choices":[{{"delta":{{"content":{json.dumps(char)}}}}}]}}\n\n'
        for char in text
    )
    body = (
        chunks +
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


def _make_handler(
    *,
    tps_mtp: float,
    tps_baseline: float,
    clock: _FakeBenchClock | None = None,
    captured_bodies: list[dict] | None = None,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            text = str(body.get("prompt") or "")
            return httpx.Response(
                200,
                json={"count": len(text), "tokens": [ord(char) for char in text]},
            )
        # Headers carry which URL was hit; we route by host.
        host = request.url.host
        tps = tps_mtp if host and host.startswith("mtp") else tps_baseline
        if clock is not None:
            clock.schedule(completion_tokens=8, tps=tps)
        if captured_bodies is not None:
            captured_bodies.append(json.loads(request.content))
        return _stream_response(text="response")

    return handler


def test_bench_command_emits_summary(monkeypatch, tmp_path):
    captured_bodies: list[dict] = []
    clock = _install_fake_clock(monkeypatch)
    handler = _make_handler(
        tps_mtp=20.0,
        tps_baseline=10.0,
        clock=clock,
        captured_bodies=captured_bodies,
    )
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--max-tokens",
            "8",
            "--runs",
            "2",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(out.read_text())
    serialized = json.dumps(body)
    assert body["profile"] == "safe80"
    assert body["output_token_target"] == 8
    assert len(body["observations"]) == 2
    assert body["statistics"]["speedup"]["median"] == pytest.approx(2.0)
    assert body["observations"][0]["mtp"]["e2e_output_tokens_per_second"] == 20.0
    assert body["observations"][0]["mtp"]["ttft_ms"] is not None
    assert body["observations"][0]["mtp"]["tpot_ms"] is not None
    assert body["observations"][0]["parity_basis"] == "token"
    assert "generation_tps" not in serialized
    assert captured_bodies
    assert all(request_body["stream"] is True for request_body in captured_bodies)
    assert all(request_body["min_tokens"] == 8 for request_body in captured_bodies)
    assert all(request_body["max_tokens"] == 8 for request_body in captured_bodies)
    assert all(request_body["ignore_eos"] is True for request_body in captured_bodies)
    assert all(request_body["temperature"] == 0.0 for request_body in captured_bodies)
    assert all(request_body["top_p"] == 1.0 for request_body in captured_bodies)
    assert body["observations"][0]["mtp_metrics_before"]["state"] == "unavailable"
    assert body["observations"][0]["mtp_metrics_delta"]["state"] == "unavailable"


def test_bench_command_records_mtp_metric_deltas(monkeypatch, tmp_path):
    metrics_values = iter([0, 8])
    clock = _install_fake_clock(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            drafted = next(metrics_values)
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    observation = json.loads(out.read_text())["observations"][0]
    assert observation["mtp_metrics_before"]["drafted_tokens_total"] == 0.0
    assert observation["mtp_metrics_after"]["drafted_tokens_total"] == 8.0
    assert observation["mtp_metrics_delta"]["state"] == "active"
    assert observation["mtp_metrics_delta"]["drafted_tokens_delta"] == 8.0
    assert observation["deterministic_parity"] is True
    assert observation["parity_basis"] == "token"
    assert observation["parity_failure"] is False


def test_measure_does_not_invent_token_latency_for_multi_token_chunks(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.e2e_output_tokens_per_second == 10.0
    assert result.ttft_ms is not None
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_leaves_throughput_unknown_when_usage_and_tokenizer_are_missing(
    monkeypatch,
):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(404, json={"error": "missing"})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens is None
    assert result.e2e_output_tokens_per_second is None
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_uses_tokenizer_count_without_inventing_token_latency(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]
    tokenize_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            tokenize_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"count": 8, "tokens": list(range(8))})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens == 8
    assert result.e2e_output_tokens_per_second == 10.0
    assert tokenize_bodies == [
        {"model": "gemma-4-31b-mtp", "prompt": "abcdefgh"}
    ]
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_tokenizer_fallback_uses_visible_text_not_chat_template(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if "messages" in body:
                return httpx.Response(
                    200,
                    json={"count": 99, "tokens": list(range(99))},
                )
            assert body == {"model": "gemma-4-31b-mtp", "prompt": "abcdefgh"}
            return httpx.Response(200, json={"count": 8})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens == 8
    assert result.e2e_output_tokens_per_second == 10.0
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_bench_command_writes_v2_artifact_directory(monkeypatch, tmp_path):
    clock = _install_fake_clock(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    "vllm:spec_decode_num_draft_tokens_total 8\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    artifact_root = tmp_path / "artifacts" / "benchmarks"
    runtime_manifest = tmp_path / "runtime-manifest.json"
    runtime_manifest.write_text(
        json.dumps({"argv": ["vllm", "serve"], "vllm_version": "0.21.0"}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--max-tokens",
            "8",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--artifact-root",
            str(artifact_root),
            "--artifact-id",
            "20260622-test-safe80",
            "--runtime-manifest-path",
            str(runtime_manifest),
        ],
    )

    artifact_dir = artifact_root / "20260622-test-safe80"
    assert result.exit_code == 0, result.stdout
    assert artifact_dir.is_dir()
    expected_files = {
        "manifest.json",
        "results.json",
        "results.md",
        "metrics-before.prom",
        "metrics-after.prom",
        "metrics-delta.json",
        "runtime-manifest.json",
        "request-payloads.json",
        "environment.txt",
        "nvidia-smi.csv",
        "README.md",
    }
    assert expected_files == {path.name for path in artifact_dir.iterdir()}
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    runtime_manifest_payload = json.loads(
        (artifact_dir / "runtime-manifest.json").read_text()
    )
    results = json.loads((artifact_dir / "results.json").read_text())
    request_payloads = json.loads((artifact_dir / "request-payloads.json").read_text())
    serialized = json.dumps(results)

    assert manifest["benchmark_protocol_version"] == 2
    assert manifest["package_version"] == "0.2.0a1"
    assert manifest["runtime_manifest_source"] == "provided"
    assert str(runtime_manifest) not in json.dumps(manifest)
    assert runtime_manifest_payload["vllm_version"] == "0.21.0"
    assert manifest["service_urls"] == {
        "mtp": "http://mtp:8000",
        "baseline": "http://baseline:8000",
    }
    assert request_payloads["chat_completion"]["stream"] is True
    assert results["statistics"]["mtp"]["e2e_output_tokens_per_second"]["median"] == 20.0
    assert "generation_tps" not in serialized


def test_bench_matrix_iterates_prompts_and_n(monkeypatch, tmp_path):
    clock = _install_fake_clock(monkeypatch)
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
            "--depth-mtp-url",
            "2=http://mtp2:8000",
            "--depth-mtp-url",
            "4=http://mtp4:8000",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert len(payload) == 4
    keys = {(entry["prompt"], entry["num_speculative_tokens"]) for entry in payload}
    assert keys == {("alpha", 2), ("alpha", 4), ("beta", 2), ("beta", 4)}
    assert all("generation_tps" not in json.dumps(entry) for entry in payload)


def test_bench_matrix_rejects_fake_multi_depth_sweep(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(_make_handler(tps_mtp=20.0, tps_baseline=10.0)),
    )

    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
        ],
    )

    assert result.exit_code == 2
    assert "single --mtp-url cannot change live vLLM speculative depth" in result.stderr


def test_bench_matrix_rejects_duplicate_depth_urls(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(_make_handler(tps_mtp=20.0, tps_baseline=10.0)),
    )

    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
            "--depth-mtp-url",
            "2=http://mtp:8000",
            "--depth-mtp-url",
            "4=http://mtp:8000",
        ],
    )

    assert result.exit_code == 2
    assert "distinct --depth-mtp-url endpoint" in result.stderr
