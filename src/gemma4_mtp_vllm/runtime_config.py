from __future__ import annotations

from typing import Any

from gemma4_mtp_vllm.profiles import ModelProfile


def desired_config(profile: ModelProfile) -> dict[str, Any]:
    return {
        "target_model": profile.target,
        "drafter": profile.drafter,
        "num_speculative_tokens": profile.num_speculative_tokens,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "max_model_len": profile.max_model_len,
        "cpu_offload_gb": profile.cpu_offload_gb,
        "max_num_seqs": profile.max_num_seqs,
        "max_num_batched_tokens": profile.max_num_batched_tokens,
        "enforce_eager": profile.enforce_eager,
        "language_model_only": profile.language_model_only,
        "validation_level": profile.validation_level,
        "max_output_tokens": profile.max_output_tokens,
        "quantization": profile.quantization,
        "kv_cache_dtype": profile.kv_cache_dtype,
    }


def observed_config_from_models(
    models_body: dict[str, Any],
    *,
    target_model: str,
    served_model_name: str | None,
) -> dict[str, Any]:
    entries = [
        entry
        for entry in models_body.get("data") or []
        if isinstance(entry, dict)
    ]
    ids = [entry.get("id") for entry in entries if isinstance(entry.get("id"), str)]
    match = _find_model_entry(entries, served_model_name) or _find_model_entry(
        entries, target_model
    )
    return {
        "models": ids,
        "served_model_name": match.get("id") if match else None,
        "target_served": target_model in ids or served_model_name in ids,
        "max_model_len": _int_or_none(match.get("max_model_len")) if match else None,
    }


def config_matches(
    desired: dict[str, Any],
    observed: dict[str, Any],
) -> bool:
    return (
        bool(observed.get("target_served"))
        and observed.get("max_model_len") == desired.get("max_model_len")
    )


def mtp_observed_from_metrics(metrics_text: str) -> bool:
    return "spec_decode" in metrics_text or "speculative" in metrics_text.lower()


def _find_model_entry(
    entries: list[dict[str, Any]],
    model_id: str | None,
) -> dict[str, Any] | None:
    if model_id is None:
        return None
    for entry in entries:
        if entry.get("id") == model_id:
            return entry
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
