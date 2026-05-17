from __future__ import annotations

from typing import Any

import httpx

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import ModelProfile
from gemma4_mtp_vllm.versioning import version_at_least


async def build_report(
    *,
    profile: ModelProfile,
    vllm_base_url: str,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    if transport is not None:
        http = httpx.AsyncClient(transport=transport, base_url=vllm_base_url)
    else:
        http = httpx.AsyncClient(base_url=vllm_base_url)
    client = VllmClient(http=http, base_url=vllm_base_url)
    try:
        return await _build_report(client=client, profile=profile)
    finally:
        await client.aclose()


async def _build_report(
    *,
    client: VllmClient,
    profile: ModelProfile,
) -> dict[str, Any]:
    vllm_status: dict[str, Any] = {"status": "unreachable", "version": None}
    target_served = False
    try:
        await client.health()
        vllm_status["status"] = "ok"
    except (VllmHttpError, httpx.HTTPError):
        vllm_status["status"] = "unreachable"

    if vllm_status.get("status") == "ok":
        try:
            version_body = await client.version()
            vllm_status["version"] = version_body.get("version")
        except (VllmHttpError, httpx.HTTPError):
            vllm_status["version"] = None
        try:
            models_body = await client.list_models()
            ids = {entry.get("id") for entry in models_body.get("data") or []}
            target_served = profile.target in ids
        except (VllmHttpError, httpx.HTTPError):
            target_served = False

    version_ok = version_at_least(
        vllm_status.get("version"),
        REQUIRED_VLLM_MIN_VERSION,
    )
    ok = vllm_status.get("status") == "ok" and version_ok and target_served
    return {
        "ok": ok,
        "profile": profile.name,
        "target_model": profile.target,
        "drafter": profile.drafter,
        "drafter_configured": profile.drafter,
        "drafter_loaded": "unknown",
        "num_speculative_tokens": profile.num_speculative_tokens,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "gateway_version": __version__,
        "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
        "vllm": vllm_status,
        "version_ok": version_ok,
        "target_served": target_served,
    }
