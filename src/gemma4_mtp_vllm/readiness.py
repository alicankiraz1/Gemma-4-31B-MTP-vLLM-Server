from __future__ import annotations

from typing import Any

from gemma4_mtp_vllm.profiles import ModelProfile


READINESS_REQUIRED_FIELDS = {
    "target_model",
    "served_model_name",
    "drafter_model",
    "mtp_method",
    "num_speculative_tokens",
    "tensor_parallel_size",
    "quantization",
    "cpu_offload_gb",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enforce_eager",
    "language_model_only",
    "gpu_memory_utilization",
}

READINESS_SEPARATE_FIELDS = {"target_served", "vllm_version"}


def build_readiness_state(
    *,
    profile: ModelProfile,
    vllm_status: dict[str, Any],
    version_ok: bool,
    target_served: bool,
    config_verification: dict[str, Any],
    mtp: dict[str, Any] | None,
    runtime: dict[str, Any],
    last_backend_error: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []

    backend_status = vllm_status.get("status")
    if backend_status == "unreachable":
        reasons.append("backend_unreachable")
        return _readiness("unavailable", reasons, warnings)

    if backend_status != "ok":
        raw_status = _raw_backend_status(vllm_status)
        if raw_status in {"starting", "loading", "initializing", "warming"}:
            reasons.append(f"backend_{raw_status}")
            return _readiness("starting", reasons, warnings)
        reasons.append("backend_not_ready")
        return _readiness("degraded", reasons, warnings)

    if not version_ok:
        reasons.append("old_vllm_version")
    if not target_served:
        reasons.append("target_not_served")

    fields = config_verification.get("fields")
    if isinstance(fields, dict):
        for field_name, field in sorted(fields.items()):
            if not isinstance(field, dict):
                continue
            if field_name in READINESS_SEPARATE_FIELDS:
                continue
            field_status = field.get("status")
            if field_status == "mismatch":
                reasons.append(f"config_mismatch:{field_name}")
            elif (
                field_status == "unknown"
                and field_name in READINESS_REQUIRED_FIELDS
            ):
                warnings.append(f"config_unknown:{field_name}")

    if last_backend_error:
        warnings.append(f"last_backend_error:{last_backend_error}")

    mtp_warning_state = _append_mtp_readiness(
        profile=profile,
        mtp=mtp,
        runtime=runtime,
        reasons=reasons,
        warnings=warnings,
    )
    if reasons:
        return _readiness("degraded", reasons, warnings)
    if mtp_warning_state == "warming":
        return _readiness("warming", reasons, warnings)
    return _readiness("ready", reasons, warnings)


def _append_mtp_readiness(
    *,
    profile: ModelProfile,
    mtp: dict[str, Any] | None,
    runtime: dict[str, Any],
    reasons: list[str],
    warnings: list[str],
) -> str | None:
    if profile.num_speculative_tokens <= 0 or profile.language_model_only:
        return None

    mtp_state = mtp.get("state") if isinstance(mtp, dict) else None
    successful_generations = _int_value(runtime.get("batch_requests"))
    last_delta = runtime.get("last_mtp_delta")

    if successful_generations == 0:
        if mtp_state in {"registered_but_idle", "not_registered"}:
            warnings.append("mtp_not_exercised")
            return "warming"
        if mtp_state in {None, "unavailable", "parse_error"}:
            warnings.append(f"mtp_metrics_{mtp_state or 'missing'}")
        return None

    delta_state = last_delta.get("state") if isinstance(last_delta, dict) else None
    if delta_state == "active":
        return None
    if delta_state in {"registered_but_idle", "not_registered"}:
        reasons.append("mtp_inactive_after_generation")
        warnings.append("mtp_inactive_after_generation")
    elif delta_state in {None, "unavailable", "parse_error"}:
        reasons.append("mtp_unverified_after_generation")
        warnings.append(f"mtp_metrics_{delta_state or 'missing'}")
    return None


def _readiness(
    state: str,
    reasons: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "state": state,
        "reasons": _dedupe(reasons),
        "warnings": _dedupe(warnings),
    }


def _raw_backend_status(vllm_status: dict[str, Any]) -> str | None:
    raw = vllm_status.get("raw")
    if not isinstance(raw, dict):
        return None
    value = raw.get("status")
    return value if isinstance(value, str) else None


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
