from __future__ import annotations

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
    assert report["config_matches"] is True
    assert report["mtp_observed"] is True
