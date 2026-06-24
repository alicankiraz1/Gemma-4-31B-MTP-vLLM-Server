from __future__ import annotations

from datetime import UTC, datetime
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from gemma4_mtp_vllm.profiles import ModelProfile
from gemma4_mtp_vllm.runtime_config import (
    argv_fingerprint,
    redact_argv,
    redact_path,
    redact_public_value,
)


def build_vllm_serve_args(
    *,
    profile: ModelProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_mtp: bool = True,
    served_model_name: str | None = None,
) -> list[str]:
    args: list[str] = [
        "vllm",
        "serve",
        profile.target,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
        "--max-model-len",
        str(profile.max_model_len),
        "--gpu-memory-utilization",
        f"{profile.gpu_memory_utilization:.2f}",
        "--cpu-offload-gb",
        _format_float(profile.cpu_offload_gb),
    ]
    if profile.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(profile.max_num_seqs)])
    if profile.max_num_batched_tokens is not None:
        args.extend(["--max-num-batched-tokens", str(profile.max_num_batched_tokens)])
    if profile.enforce_eager:
        args.append("--enforce-eager")
    if profile.language_model_only:
        args.append("--language-model-only")
    args.extend(["--reasoning-parser", "gemma4"])
    if profile.quantization is not None:
        args.extend(["--quantization", profile.quantization])
    if profile.kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", profile.kv_cache_dtype])
    if served_model_name:
        args.extend(["--served-model-name", served_model_name])
    if enable_mtp:
        spec = {
            "method": "mtp",
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec, separators=(",", ":"))])
    return args


def build_launch_manifest(
    *,
    profile: ModelProfile,
    argv: list[str],
    enable_mtp: bool,
    served_model_name: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "cwd": redact_path(str(Path.cwd())),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty": _git_dirty(),
        "profile": profile.name,
        "served_model_name": redact_public_value(served_model_name),
        "enable_mtp": enable_mtp,
        "stream_interval_control": "unavailable",
        "argv": redact_argv(argv),
        "argv_fingerprint": argv_fingerprint(argv),
        "package_versions": {
            name: _package_version(name)
            for name in (
                "gemma4-mtp-vllm",
                "vllm",
                "torch",
                "fastapi",
                "starlette",
                "prometheus-fastapi-instrumentator",
            )
        },
    }


def write_launch_manifest(
    *,
    path: Path,
    profile: ModelProfile,
    argv: list[str],
    enable_mtp: bool,
    served_model_name: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_launch_manifest(
                profile=profile,
                argv=argv,
                enable_mtp=enable_mtp,
                served_model_name=served_model_name,
            ),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_vllm_executable(command: str = "vllm") -> str:
    found = shutil.which(command)
    if found is not None:
        return found
    sibling = Path(sys.executable).with_name(command)
    if sibling.exists():
        return str(sibling)
    return command


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(status.strip())
