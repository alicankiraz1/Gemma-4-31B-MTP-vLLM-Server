from __future__ import annotations

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from gemma4_mtp_vllm.cli import app


runner = CliRunner()


def _make_handler(*, tps_mtp: float, tps_baseline: float):
    def handler(request: httpx.Request) -> httpx.Response:
        # Headers carry which URL was hit; we route by host.
        host = request.url.host
        tps = tps_mtp if host and host.startswith("mtp") else tps_baseline
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "vllm_tps_for_test": tps,
            },
        )

    return handler


def test_bench_command_emits_summary(monkeypatch, tmp_path):
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0)
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
    assert body["profile"] == "safe80"
    assert len(body["observations"]) == 2
    assert body["median_speedup"] == 2.0


def test_bench_matrix_iterates_prompts_and_n(monkeypatch, tmp_path):
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0)
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
