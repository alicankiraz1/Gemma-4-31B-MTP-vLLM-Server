from __future__ import annotations

import json

import httpx
import pytest

from gemma4_mtp_vllm.doctor import build_report
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


@pytest.mark.asyncio
async def test_doctor_ok_when_target_served_and_drafter_not_listed():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "google/gemma-4-31B-it"},
                ],
            })
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is True
    assert report["profile"] == "safe80"
    assert report["target_model"] == "google/gemma-4-31B-it"
    assert report["drafter"] == "google/gemma-4-31B-it-assistant"
    assert report["vllm"]["status"] == "ok"
    assert report["vllm"]["version"] == "0.21.0"
    assert report["version_ok"] is True
    assert report["target_served"] is True
    assert report["drafter_configured"] == "google/gemma-4-31B-it-assistant"
    assert report["drafter_loaded"] == "unknown"


@pytest.mark.asyncio
async def test_doctor_rejects_old_vllm_version():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.20.2"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "google/gemma-4-31B-it"},
                ],
            })
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["version_ok"] is False


@pytest.mark.asyncio
async def test_doctor_rejects_profile_name_as_served_target():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "safe80"},
                ],
            })
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["target_served"] is False


@pytest.mark.asyncio
async def test_doctor_ok_when_served_model_name_listed():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "gemma-4-31b-mtp"},
                ],
            })
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
        served_model_name="gemma-4-31b-mtp",
    )
    assert report["ok"] is True
    assert report["target_served"] is True
    assert report["served_model_name"] == "gemma-4-31b-mtp"


@pytest.mark.asyncio
async def test_doctor_marks_not_ok_when_target_missing():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["target_served"] is False


@pytest.mark.asyncio
async def test_doctor_marks_not_ok_when_vllm_unreachable():
    def handler(request):
        raise httpx.ConnectError("nope")

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["vllm"]["status"] == "unreachable"


@pytest.mark.asyncio
async def test_doctor_reports_observed_config_and_mtp_metric_separately():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "gemma-4-31b-mtp", "max_model_len": 2048},
                ],
            })
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="vllm:spec_decode_draft_acceptance_rate 0.58\n",
            )
        return httpx.Response(404)

    report = await build_report(
        profile=resolve_profile("tp2_2x32_smoke", load_profiles()),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
        served_model_name="gemma-4-31b-mtp",
    )

    assert report["ok"] is True
    assert report["observed_config"]["max_model_len"] == 2048
    assert report["config_verification"]["status"] == "partial"
    assert report["config_verification"]["fields"]["quantization"]["status"] == "not_applicable"
    assert report["config_verification"]["fields"]["cpu_offload_gb"]["status"] == "unknown"
    assert report["config_matches"] is False
    assert report["mtp_observed"] is True


@pytest.mark.asyncio
async def test_doctor_verifies_runtime_fields_from_active_manifest(tmp_path):
    profile = resolve_profile("tp2_2x32_fp8_gpuonly", load_profiles())
    argv = [
        "vllm",
        "serve",
        "google/gemma-4-31B-it",
        "--served-model-name",
        "gemma-4-31b-mtp",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.95",
        "--cpu-offload-gb",
        "0",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
        "--quantization",
        "fp8",
        "--api-key",
        "secret-token",
        "--speculative-config",
        '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pid": 321,
                "argv": argv,
                "package_versions": {"torch": "2.11.0+cu130", "vllm": "0.21.0"},
            }
        ),
        encoding="utf-8",
    )

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "gemma-4-31b-mtp", "max_model_len": 2048}]},
            )
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="vllm:spec_decode_draft_acceptance_rate 0.58\n",
            )
        return httpx.Response(404)

    report = await build_report(
        profile=profile,
        vllm_base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
        served_model_name="gemma-4-31b-mtp",
        runtime_manifest_path=manifest,
        active_backend_pid=321,
        active_backend_argv=argv,
    )

    assert report["config_verification"]["status"] == "partial"
    assert report["config_verification"]["fields"]["cpu_offload_gb"]["status"] == "verified"
    assert report["config_verification"]["fields"]["quantization"]["status"] == "verified"
    assert report["observed_config"]["runtime_argv"][argv.index("secret-token")] == "REDACTED"
    assert "secret-token" not in json.dumps(report)


@pytest.mark.asyncio
async def test_doctor_reports_cpu_offload_mismatch_from_active_manifest(tmp_path):
    profile = resolve_profile("tp2_2x32_fp8_gpuonly", load_profiles())
    argv = [
        "vllm",
        "serve",
        "google/gemma-4-31B-it",
        "--served-model-name",
        "gemma-4-31b-mtp",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.95",
        "--cpu-offload-gb",
        "8",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
        "--quantization",
        "fp8",
        "--speculative-config",
        '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pid": 321, "argv": argv}), encoding="utf-8")

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "gemma-4-31b-mtp", "max_model_len": 2048}]},
            )
        return httpx.Response(404)

    report = await build_report(
        profile=profile,
        vllm_base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
        served_model_name="gemma-4-31b-mtp",
        runtime_manifest_path=manifest,
        active_backend_pid=321,
        active_backend_argv=argv,
    )

    assert report["config_verification"]["status"] == "mismatch"
    assert report["config_verification"]["fields"]["cpu_offload_gb"]["status"] == "mismatch"
    assert report["config_matches"] is False


@pytest.mark.asyncio
async def test_doctor_redacts_private_served_model_name():
    private_name = "/" + "home" + "/homelander/private-served-name"

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": private_name}]})
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
        served_model_name=private_name,
    )

    assert report["served_model_name"] == "REDACTED_PATH"
    assert private_name not in json.dumps(report)
