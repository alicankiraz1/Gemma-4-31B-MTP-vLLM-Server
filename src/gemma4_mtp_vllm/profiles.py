from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "profiles.yaml"
)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    target: str
    drafter: str
    num_speculative_tokens: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    temperature: float
    top_p: float
    top_k: int
    requires_vram_gb: int
    cpu_offload_gb: float = 0.0
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False
    language_model_only: bool = False
    validation_level: str = "unverified"
    max_output_tokens: int = 4096
    quantization: str | None = None
    kv_cache_dtype: str | None = None


@dataclass(frozen=True)
class ProfileSet:
    default: str
    aliases: dict[str, str]
    items: dict[str, ModelProfile]


def load_profiles(path: Path | None = None) -> ProfileSet:
    profiles_file = _profiles_file(path)
    with profiles_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    default = _required_str(raw, "default")
    aliases = dict(raw.get("aliases") or {})
    profile_items = raw.get("profiles") or {}
    if not isinstance(profile_items, dict):
        raise ValueError("profiles must be a mapping")

    items = {
        name: ModelProfile(name=name, **_profile_fields(config))
        for name, config in profile_items.items()
    }

    if default not in items:
        raise ValueError(f"default profile {default!r} does not exist")

    missing_aliases = {
        alias: profile_name
        for alias, profile_name in aliases.items()
        if profile_name not in items
    }
    if missing_aliases:
        raise ValueError(f"aliases point to unknown profiles: {missing_aliases!r}")

    return ProfileSet(default=default, aliases=aliases, items=items)


def _profiles_file(path: Path | None) -> Any:
    if path is not None:
        return Path(path)

    package_profiles = resources.files(__package__).joinpath("config/profiles.yaml")
    if package_profiles.is_file():
        return package_profiles

    if DEFAULT_PROFILES_PATH.exists():
        return DEFAULT_PROFILES_PATH
    return DEFAULT_PROFILES_PATH


def resolve_profile(name: str | None, profiles: ProfileSet) -> ModelProfile:
    profile_name = profiles.default if name is None else profiles.aliases.get(name, name)
    try:
        return profiles.items[profile_name]
    except KeyError as exc:
        raise KeyError(f"unknown profile or alias: {name!r}") from exc


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _profile_fields(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("profile entries must be mappings")

    required_fields = {
        "target": str,
        "drafter": str,
        "num_speculative_tokens": int,
        "tensor_parallel_size": int,
        "gpu_memory_utilization": (int, float),
        "max_model_len": int,
        "temperature": (int, float),
        "top_p": (int, float),
        "top_k": int,
        "requires_vram_gb": int,
    }
    optional_fields: dict[str, tuple[type | tuple[type, ...], Any]] = {
        "cpu_offload_gb": ((int, float), 0.0),
        "max_num_seqs": (int, None),
        "max_num_batched_tokens": (int, None),
        "enforce_eager": (bool, False),
        "language_model_only": (bool, False),
        "validation_level": (str, "unverified"),
        "max_output_tokens": (int, 4096),
        "quantization": (str, None),
        "kv_cache_dtype": (str, None),
    }

    values: dict[str, Any] = {}
    for key, expected_type in required_fields.items():
        value = config.get(key)
        if isinstance(value, bool) and _expects_int(expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if not isinstance(value, expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if key in {"gpu_memory_utilization", "temperature", "top_p"}:
            values[key] = float(value)
        else:
            values[key] = value

    for key, (expected_type, default) in optional_fields.items():
        value = config.get(key, default)
        if value is None:
            values[key] = None
            continue
        if isinstance(value, bool) and _expects_int(expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if not isinstance(value, expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if key == "cpu_offload_gb":
            values[key] = float(value)
        else:
            values[key] = value

    if values["num_speculative_tokens"] <= 0:
        raise ValueError("num_speculative_tokens must be positive")
    if values["tensor_parallel_size"] <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not 0.0 < values["gpu_memory_utilization"] <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if values["max_model_len"] <= 0:
        raise ValueError("max_model_len must be positive")
    if not 0.0 <= values["top_p"] <= 1.0:
        raise ValueError("top_p must be between 0 and 1")
    if values["temperature"] < 0.0:
        raise ValueError("temperature must be non-negative")
    if values["top_k"] < 0:
        raise ValueError("top_k must be non-negative")
    if values["requires_vram_gb"] <= 0:
        raise ValueError("requires_vram_gb must be positive")
    if values["cpu_offload_gb"] < 0:
        raise ValueError("cpu_offload_gb must be non-negative")
    if values["max_num_seqs"] is not None and values["max_num_seqs"] <= 0:
        raise ValueError("max_num_seqs must be positive")
    if (
        values["max_num_batched_tokens"] is not None
        and values["max_num_batched_tokens"] <= 0
    ):
        raise ValueError("max_num_batched_tokens must be positive")
    if values["max_output_tokens"] <= 0:
        raise ValueError("max_output_tokens must be positive")
    if values["validation_level"] not in {"unverified", "smoke", "validated"}:
        raise ValueError("validation_level must be unverified, smoke, or validated")
    for key in ("quantization", "kv_cache_dtype"):
        value = values[key]
        if value is not None and not value.strip():
            raise ValueError(f"{key} must be non-empty when set")

    return values


def _expects_int(expected_type: type | tuple[type, ...]) -> bool:
    if expected_type is int:
        return True
    return isinstance(expected_type, tuple) and int in expected_type
