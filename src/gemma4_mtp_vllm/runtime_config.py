from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION
from gemma4_mtp_vllm.graph_observation import parse_cuda_graph_observation
from gemma4_mtp_vllm.mtp_metrics import parse_mtp_metrics
from gemma4_mtp_vllm.profiles import ModelProfile
from gemma4_mtp_vllm.versioning import version_at_least


ATTESTATION_FIELDS = (
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
    "kv_cache_dtype",
    "vllm_version",
    "torch_version",
    "cuda_version",
    "target_served",
)

OPTIONAL_PROFILE_FIELDS = {
    "kv_cache_dtype",
    "max_num_seqs",
    "max_num_batched_tokens",
    "quantization",
}

SECRET_VALUE_RE = re.compile(
    r"(hf_[A-Za-z0-9_]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})"
)
PRIVATE_PATH_RE = re.compile(r"/(?:Users|home)/[^\s\"',}\]]+")
SECRET_OPTIONS = {
    "--api-key",
    "--token",
    "--hf-token",
    "--hugging-face-token",
    "--password",
}
PRIVATE_PATH_PREFIXES = ("/" "Users" "/", "/" "home" "/")
LOG_TAIL_BYTES = 256 * 1024


def desired_config(
    profile: ModelProfile,
    *,
    served_model_name: str | None = None,
) -> dict[str, Any]:
    return {
        "target_model": redact_public_value(profile.target),
        "drafter": redact_public_value(profile.drafter),
        "drafter_model": redact_public_value(profile.drafter),
        "served_model_name": redact_public_value(served_model_name),
        "mtp_method": "mtp",
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
        "vllm_version": REQUIRED_VLLM_MIN_VERSION,
        "torch_version": None,
        "cuda_version": None,
        "target_served": True,
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
    observed = {
        "models": [redact_public_value(model_id) for model_id in ids],
        "target_model": redact_public_value(target_model) if target_model in ids else None,
        "served_model_name": redact_public_value(match.get("id")) if match else None,
        "target_served": target_model in ids or served_model_name in ids,
        "max_model_len": _int_or_none(match.get("max_model_len")) if match else None,
    }
    sources: dict[str, str] = {}
    if observed["target_model"] is not None:
        sources["target_model"] = "vllm_models_api"
    if observed["served_model_name"] is not None:
        sources["served_model_name"] = "vllm_models_api"
    sources["target_served"] = "vllm_models_api"
    if observed["max_model_len"] is not None:
        sources["max_model_len"] = "vllm_models_api"
    observed["_sources"] = sources
    return observed


def config_matches(
    desired: dict[str, Any],
    observed: dict[str, Any],
) -> bool:
    return build_config_verification(desired, observed)["status"] == "verified"


def merge_observed_config(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for config in configs:
        for key, value in config.items():
            if key == "_sources" and isinstance(value, dict):
                sources.update(value)
                continue
            if value is None and merged.get(key) is not None:
                continue
            merged[key] = value
    if sources:
        merged["_sources"] = sources
    return merged


def observed_config_from_version(version: str | None) -> dict[str, Any]:
    if not version:
        return {}
    return {"vllm_version": version, "_sources": {"vllm_version": "vllm_version_api"}}


def observed_config_from_metrics(
    metrics_text: str,
    *,
    model_name: str | None = None,
    log_text: str = "",
) -> dict[str, Any]:
    mtp = parse_mtp_metrics(metrics_text, model_name=model_name)
    cuda_graph = parse_cuda_graph_observation(
        metrics_text=metrics_text,
        log_text=log_text,
    )
    return {
        "mtp": mtp,
        "mtp_observed": mtp.get("state") == "active",
        "cuda_graph": cuda_graph,
        "cuda_graph_observed": cuda_graph.get("graph_active"),
        "_sources": {
            "mtp": "vllm_metrics",
            "mtp_observed": "vllm_metrics",
            "cuda_graph": _cuda_graph_source(cuda_graph),
            "cuda_graph_observed": _cuda_graph_source(cuda_graph),
        },
    }


def default_observed_config() -> dict[str, Any]:
    return {
        "models": [],
        "served_model_name": None,
        "target_served": False,
        "max_model_len": None,
        "mtp_observed": False,
        "_sources": {
            "target_served": "unknown",
            "mtp_observed": "unknown",
        },
    }


def build_config_verification(
    desired: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    sources = observed.get("_sources") if isinstance(observed.get("_sources"), dict) else {}
    fields = {
        field: _verification_field(
            field,
            desired=_desired_field_value(desired, field),
            observed=observed.get(field),
            source=sources.get(field, "unknown"),
        )
        for field in ATTESTATION_FIELDS
    }
    statuses = [field["status"] for field in fields.values()]
    if "mismatch" in statuses:
        status = "mismatch"
    elif all(value in {"verified", "not_applicable"} for value in statuses):
        status = "verified"
    elif "verified" in statuses:
        status = "partial"
    else:
        status = "unknown"
    return {"status": status, "fields": fields}


def public_observed_config(observed: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observed.items() if key != "_sources"}


def observed_config_from_process_argv(argv: list[str]) -> dict[str, Any]:
    observed: dict[str, Any] = {"runtime_argv": redact_argv(argv)}
    sources: dict[str, str] = {"runtime_argv": "process_cmdline"}

    serve_index = _find_serve_index(argv)
    if serve_index is not None and serve_index + 1 < len(argv):
        observed["target_model"] = redact_public_value(argv[serve_index + 1])
        sources["target_model"] = "process_cmdline"

    option_map = {
        "served_model_name": "--served-model-name",
        "tensor_parallel_size": "--tensor-parallel-size",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "cpu_offload_gb": "--cpu-offload-gb",
        "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "quantization": "--quantization",
        "kv_cache_dtype": "--kv-cache-dtype",
    }
    for field, option in option_map.items():
        value = _option_value(argv, option)
        if value is None:
            continue
        observed[field] = _coerce_argv_value(field, value)
        sources[field] = "process_cmdline"

    observed["enforce_eager"] = "--enforce-eager" in argv
    sources["enforce_eager"] = "process_cmdline"
    observed["language_model_only"] = "--language-model-only" in argv
    sources["language_model_only"] = "process_cmdline"

    speculative_config = _option_value(argv, "--speculative-config")
    if speculative_config:
        try:
            spec = json.loads(speculative_config)
        except json.JSONDecodeError:
            spec = {}
        if isinstance(spec, dict):
            if "method" in spec:
                observed["mtp_method"] = spec["method"]
                sources["mtp_method"] = "process_cmdline"
            if "model" in spec:
                observed["drafter_model"] = (
                    redact_public_value(spec["model"]) if isinstance(spec["model"], str) else spec["model"]
                )
                sources["drafter_model"] = "process_cmdline"
            if "num_speculative_tokens" in spec:
                observed["num_speculative_tokens"] = _int_or_none(
                    spec["num_speculative_tokens"]
                )
                sources["num_speculative_tokens"] = "process_cmdline"

    observed["_sources"] = sources
    return observed


def observed_config_from_active_manifest(
    manifest: dict[str, Any],
    *,
    active_pid: int | None,
    process_argv: list[str] | None,
) -> dict[str, Any]:
    manifest_pid = _int_or_none(manifest.get("pid"))
    manifest_argv = manifest.get("argv")
    manifest_fingerprint = manifest.get("argv_fingerprint")
    if (
        manifest_pid is None
        or active_pid != manifest_pid
        or not isinstance(manifest_argv, list)
        or process_argv is None
        or not _argv_matches_manifest(
            manifest_argv,
            process_argv,
            manifest_fingerprint=manifest_fingerprint if isinstance(manifest_fingerprint, str) else None,
        )
    ):
        return {}

    observed = observed_config_from_process_argv(process_argv)
    sources = dict(observed.get("_sources", {}))
    package_versions = manifest.get("package_versions")
    if isinstance(package_versions, dict):
        version_map = {
            "torch": "torch_version",
            "vllm": "vllm_version",
        }
        for package_name, field in version_map.items():
            value = package_versions.get(package_name)
            if isinstance(value, str) and value:
                observed[field] = value
                sources[field] = "runtime_manifest"
    observed["runtime_argv"] = redact_argv(process_argv)
    sources["runtime_argv"] = "process_cmdline"
    observed["_sources"] = sources
    return observed


def observed_config_from_runtime_evidence(
    *,
    runtime_manifest: dict[str, Any] | None = None,
    runtime_manifest_path: Path | None = None,
    active_backend_pid: int | None = None,
    active_backend_argv: list[str] | None = None,
    vllm_base_url: str | None = None,
) -> dict[str, Any]:
    manifest = runtime_manifest or read_runtime_manifest(runtime_manifest_path)
    if not manifest:
        return {}
    manifest_pid = _int_or_none(manifest.get("pid"))
    if active_backend_pid is None:
        active_backend_pid = manifest_pid
    if active_backend_argv is None and manifest_pid is not None:
        active_backend_argv = process_argv_for_pid(manifest_pid)
    if (
        active_backend_argv is not None
        and vllm_base_url is not None
        and not _argv_matches_base_url(active_backend_argv, vllm_base_url)
    ):
        return {}
    return observed_config_from_active_manifest(
        manifest,
        active_pid=active_backend_pid,
        process_argv=active_backend_argv,
    )


def read_runtime_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


def read_text_tail(path: Path | None, *, max_bytes: int = LOG_TAIL_BYTES) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            raw = handle.read(max_bytes)
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def process_argv_for_pid(pid: int) -> list[str] | None:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return None
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            redacted.append("REDACTED")
            redact_next = False
            continue
        if value in SECRET_OPTIONS:
            redacted.append(value)
            redact_next = True
            continue
        option, sep, option_value = value.partition("=")
        if sep and option in SECRET_OPTIONS:
            redacted.append(f"{option}=REDACTED")
            continue
        if sep and _contains_private_path(option_value):
            redacted.append(f"{option}=REDACTED_PATH")
            continue
        if _looks_private_path(value):
            redacted.append("REDACTED_PATH")
            continue
        redacted.append(_redact_string(value))
    return redacted


def redact_path(value: str) -> str:
    if _contains_private_path(value):
        return "REDACTED_PATH"
    return _redact_string(value)


def redact_public_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_path(value)
    if isinstance(value, list):
        return [redact_public_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_public_value(item) for key, item in value.items()}
    return value


def argv_fingerprint(argv: list[str]) -> str:
    encoded = "\0".join(argv).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def mtp_observed_from_metrics(metrics_text: str) -> bool:
    return parse_mtp_metrics(metrics_text).get("state") == "active"


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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _cuda_graph_source(cuda_graph: dict[str, Any]) -> str:
    evidence_sources = cuda_graph.get("evidence_sources")
    if not isinstance(evidence_sources, list):
        return "unknown"
    sources = {source for source in evidence_sources if isinstance(source, str)}
    if sources == {"metrics", "logs"}:
        return "vllm_metrics+vllm_logs"
    if sources == {"metrics"}:
        return "vllm_metrics"
    if sources == {"logs"}:
        return "vllm_logs"
    return "unknown"


def _desired_field_value(desired: dict[str, Any], field: str) -> Any:
    return desired.get(field)


def _verification_field(
    field: str,
    *,
    desired: Any,
    observed: Any,
    source: str,
) -> dict[str, Any]:
    result = {
        "desired": desired,
        "observed": observed,
        "status": "unknown",
        "source": source,
    }
    if desired is None and observed is None and field in OPTIONAL_PROFILE_FIELDS:
        result["status"] = "not_applicable"
        result["source"] = "profile"
        result["reason"] = "not requested by profile"
        return result
    if observed is None:
        result["reason"] = "not observed from runtime"
        return result
    if source == "unknown":
        result["reason"] = "observation source unknown"
        return result
    if desired == "REDACTED_PATH" or observed == "REDACTED_PATH":
        result["status"] = "unknown"
        result["reason"] = "private path redacted"
        return result
    if field == "vllm_version":
        result["status"] = "verified" if version_at_least(observed, desired) else "mismatch"
        return result
    if desired is None:
        result["status"] = "verified"
        return result
    result["status"] = "verified" if _values_equal(desired, observed) else "mismatch"
    return result


def _values_equal(desired: Any, observed: Any) -> bool:
    if isinstance(desired, bool):
        return observed is desired
    if isinstance(desired, int) and not isinstance(desired, bool):
        return _int_or_none(observed) == desired
    if isinstance(desired, float):
        observed_float = _float_or_none(observed)
        return observed_float is not None and abs(observed_float - desired) < 1e-9
    return observed == desired


def _find_serve_index(argv: list[str]) -> int | None:
    for index, value in enumerate(argv):
        if value == "serve":
            return index
    return None


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _coerce_argv_value(field: str, value: str) -> Any:
    if field in {
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    }:
        return _int_or_none(value)
    if field in {"gpu_memory_utilization", "cpu_offload_gb"}:
        return _float_or_none(value)
    return value


def _argv_matches_manifest(
    manifest_argv: list[Any],
    process_argv: list[str],
    *,
    manifest_fingerprint: str | None,
) -> bool:
    if manifest_fingerprint is not None:
        return manifest_fingerprint == argv_fingerprint(process_argv)
    manifest_tail = _argv_tail_from_serve(redact_argv([str(value) for value in manifest_argv]))
    process_tail = _argv_tail_from_serve(redact_argv(process_argv))
    return manifest_tail is not None and manifest_tail == process_tail


def _argv_tail_from_serve(argv: list[str]) -> list[str] | None:
    serve_index = _find_serve_index(argv)
    if serve_index is None:
        return None
    return argv[serve_index:]


def _argv_matches_base_url(argv: list[str], vllm_base_url: str) -> bool:
    parsed = urlparse(vllm_base_url)
    if parsed.hostname not in {None, "127.0.0.1", "localhost", "::1"}:
        return False
    expected_port = parsed.port
    if expected_port is None:
        expected_port = 443 if parsed.scheme == "https" else 80
    argv_port = _int_or_none(_option_value(argv, "--port") or "8000")
    return argv_port == expected_port


def _looks_private_path(value: str) -> bool:
    return value.startswith(PRIVATE_PATH_PREFIXES)


def _contains_private_path(value: str) -> bool:
    return PRIVATE_PATH_RE.search(value) is not None


def _redact_string(value: str) -> str:
    parsed = _redact_json_string(value)
    if parsed is not None:
        return parsed
    return PRIVATE_PATH_RE.sub("REDACTED_PATH", SECRET_VALUE_RE.sub("REDACTED", value))


def _redact_json_string(value: str) -> str | None:
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return json.dumps(redact_public_value(parsed), separators=(",", ":"))
