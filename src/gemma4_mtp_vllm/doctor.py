from __future__ import annotations

from typing import Any

import httpx

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import ModelProfile
from gemma4_mtp_vllm.runtime_config import (
    config_matches,
    desired_config,
    mtp_observed_from_metrics,
    observed_config_from_models,
)
from gemma4_mtp_vllm.versioning import version_at_least


async def build_report(
    *,
    profile: ModelProfile,
    vllm_base_url: str,
    transport: httpx.BaseTransport | None = None,
    served_model_name: str | None = None,
) -> dict[str, Any]:
    if transport is not None:
        http = httpx.AsyncClient(transport=transport, base_url=vllm_base_url)
    else:
        http = httpx.AsyncClient(base_url=vllm_base_url)
    client = VllmClient(http=http, base_url=vllm_base_url)
    try:
        return await _build_report(
            client=client,
            profile=profile,
            served_model_name=served_model_name,
        )
    finally:
        await client.aclose()


async def _build_report(
    *,
    client: VllmClient,
    profile: ModelProfile,
    served_model_name: str | None = None,
) -> dict[str, Any]:
    vllm_status: dict[str, Any] = {"status": "unreachable", "version": None}
    target_served = False
    desired = desired_config(profile)
    observed: dict[str, Any] = {
        "models": [],
        "served_model_name": None,
        "target_served": False,
        "max_model_len": None,
        "mtp_observed": False,
    }
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
            observed.update(
                observed_config_from_models(
                    models_body,
                    target_model=profile.target,
                    served_model_name=served_model_name,
                )
            )
            target_served = bool(observed["target_served"])
        except (VllmHttpError, httpx.HTTPError):
            target_served = False
        try:
            observed["mtp_observed"] = mtp_observed_from_metrics(
                await client.metrics_text()
            )
        except (VllmHttpError, httpx.HTTPError):
            observed["mtp_observed"] = False

    version_ok = version_at_least(
        vllm_status.get("version"),
        REQUIRED_VLLM_MIN_VERSION,
    )
    matches = config_matches(desired, observed)
    ok = vllm_status.get("status") == "ok" and version_ok and target_served
    return {
        "ok": ok,
        "profile": profile.name,
        "target_model": profile.target,
        "served_model_name": served_model_name,
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
        "desired_config": desired,
        "observed_config": observed,
        "config_matches": matches,
        "mtp_observed": observed.get("mtp_observed", False),
    }
