from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

import httpx
from typer.testing import CliRunner

from gemma4_mtp_vllm.cli import app


runner = CliRunner()


def test_launch_command_prints_argv():
    result = runner.invoke(app, ["launch", "--print-only"])
    assert result.exit_code == 0
    assert "vllm" in result.stdout
    assert "serve" in result.stdout
    assert "google/gemma-4-31B-it" in result.stdout
    assert "--served-model-name gemma-4-31b-mtp" in result.stdout


def test_launch_command_prints_shell_safe_mtp_argv():
    result = runner.invoke(app, ["launch", "--profile", "safe80", "--print-only"])
    assert result.exit_code == 0
    assert "--speculative-config" in result.stdout
    assert '"method":"mtp"' in result.stdout

    parsed = shlex.split(result.stdout.strip())
    spec_idx = parsed.index("--speculative-config")
    spec = json.loads(parsed[spec_idx + 1])
    assert spec["method"] == "mtp"


def test_launch_command_prints_2x32_smoke_args():
    result = runner.invoke(
        app,
        ["launch", "--profile", "tp2_2x32_smoke", "--port", "8000", "--print-only"],
    )

    assert result.exit_code == 0
    assert "--max-model-len 2048" in result.stdout
    assert "--cpu-offload-gb 8" in result.stdout
    assert "--max-num-seqs 1" in result.stdout
    assert "--max-num-batched-tokens 4096" in result.stdout
    assert "--enforce-eager" in result.stdout


def test_launch_rejects_public_raw_vllm_without_explicit_allow():
    result = runner.invoke(app, ["launch", "--host", "0.0.0.0", "--print-only"])
    assert result.exit_code != 0
    assert "--allow-public-vllm" in result.stdout or "--allow-public-vllm" in result.stderr


def test_launch_allows_public_raw_vllm_with_explicit_flag():
    result = runner.invoke(
        app,
        ["launch", "--host", "0.0.0.0", "--allow-public-vllm", "--print-only"],
    )
    assert result.exit_code == 0
    assert "--host 0.0.0.0" in result.stdout


def test_launch_writes_runtime_manifest_before_exec(monkeypatch, tmp_path):
    manifest = tmp_path / "launch" / "manifest.json"

    def fake_execvp(_program, _argv):
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr("gemma4_mtp_vllm.cli.os.execvp", fake_execvp)
    result = runner.invoke(
        app,
        [
            "launch",
            "--profile",
            "tp2_2x32_smoke",
            "--manifest-path",
            str(manifest),
        ],
    )

    assert result.exit_code != 0
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["profile"] == "tp2_2x32_smoke"
    assert payload["argv"][0:3] == ["vllm", "serve", "google/gemma-4-31B-it"]


def test_doctor_command_emits_json(monkeypatch):
    def fake_run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "google/gemma-4-31B-it"},
                ],
            })
        return httpx.Response(404)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    result = runner.invoke(
        app,
        ["doctor", "--profile", "safe80", "--vllm-base-url", "http://vllm.local:8000"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["version_ok"] is True
    assert payload["target_served"] is True
    assert payload["drafter_loaded"] == "unknown"


def test_serve_command_rejects_non_loopback_without_key():
    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0"],
    )
    assert result.exit_code != 0
    assert "api-key" in result.stdout.lower() or "api-key" in result.stderr.lower()
