from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import ModelProfile
from gemma4_mtp_vllm.runtime_config import (
    build_config_verification,
    config_matches,
    default_observed_config,
    desired_config,
    merge_observed_config,
    observed_config_from_models,
    observed_config_from_metrics,
    observed_config_from_runtime_evidence,
    observed_config_from_version,
    public_observed_config,
    redact_public_value,
)
from gemma4_mtp_vllm.versioning import version_at_least


async def build_report(
    *,
    profile: ModelProfile,
    vllm_base_url: str,
    transport: httpx.BaseTransport | None = None,
    served_model_name: str | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    active_backend_pid: int | None = None,
    active_backend_argv: list[str] | None = None,
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
            runtime_manifest_path=runtime_manifest_path,
            runtime_manifest=runtime_manifest,
            active_backend_pid=active_backend_pid,
            active_backend_argv=active_backend_argv,
        )
    finally:
        await client.aclose()


async def _build_report(
    *,
    client: VllmClient,
    profile: ModelProfile,
    served_model_name: str | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    active_backend_pid: int | None = None,
    active_backend_argv: list[str] | None = None,
) -> dict[str, Any]:
    vllm_status: dict[str, Any] = {"status": "unreachable", "version": None}
    target_served = False
    desired = desired_config(profile, served_model_name=served_model_name)
    observed = merge_observed_config(
        default_observed_config(),
        observed_config_from_runtime_evidence(
            runtime_manifest=runtime_manifest,
            runtime_manifest_path=runtime_manifest_path,
            active_backend_pid=active_backend_pid,
            active_backend_argv=active_backend_argv,
            vllm_base_url=client.base_url,
        ),
    )
    try:
        await client.health()
        vllm_status["status"] = "ok"
    except (VllmHttpError, httpx.HTTPError):
        vllm_status["status"] = "unreachable"

    if vllm_status.get("status") == "ok":
        try:
            version_body = await client.version()
            vllm_status["version"] = version_body.get("version")
            observed = merge_observed_config(
                observed,
                observed_config_from_version(vllm_status["version"]),
            )
        except (VllmHttpError, httpx.HTTPError):
            vllm_status["version"] = None
        try:
            models_body = await client.list_models()
            observed = merge_observed_config(
                observed,
                observed_config_from_models(
                    models_body,
                    target_model=profile.target,
                    served_model_name=served_model_name,
                ),
            )
            target_served = bool(observed["target_served"])
        except (VllmHttpError, httpx.HTTPError):
            target_served = False
        try:
            observed = merge_observed_config(
                observed,
                observed_config_from_metrics(
                    await client.metrics_text(),
                    model_name=served_model_name,
                ),
            )
        except (VllmHttpError, httpx.HTTPError):
            observed = merge_observed_config(
                observed,
                observed_config_from_metrics("", model_name=served_model_name),
            )

    version_ok = version_at_least(
        vllm_status.get("version"),
        REQUIRED_VLLM_MIN_VERSION,
    )
    matches = config_matches(desired, observed)
    verification = build_config_verification(desired, observed)
    ok = vllm_status.get("status") == "ok" and version_ok and target_served
    public_observed = public_observed_config(observed)
    return {
        "ok": ok,
        "profile": profile.name,
        "target_model": redact_public_value(profile.target),
        "served_model_name": redact_public_value(served_model_name),
        "drafter": redact_public_value(profile.drafter),
        "drafter_configured": redact_public_value(profile.drafter),
        "drafter_loaded": "unknown",
        "num_speculative_tokens": profile.num_speculative_tokens,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "gateway_version": __version__,
        "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
        "vllm": vllm_status,
        "version_ok": version_ok,
        "target_served": target_served,
        "desired_config": desired,
        "observed_config": public_observed,
        "config_verification": verification,
        "config_matches": matches,
        "mtp": public_observed.get("mtp"),
        "mtp_observed": public_observed.get("mtp_observed", False),
    }
