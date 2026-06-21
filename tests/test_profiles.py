from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

from gemma4_mtp_vllm.profiles import (
    ModelProfile,
    ProfileSet,
    load_profiles,
    resolve_profile,
)

ROOT_PROFILES = Path(__file__).resolve().parents[1] / "config" / "profiles.yaml"
PACKAGED_PROFILES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "gemma4_mtp_vllm"
    / "config"
    / "profiles.yaml"
)


def test_root_and_packaged_profiles_match():
    assert filecmp.cmp(ROOT_PROFILES, PACKAGED_PROFILES, shallow=False)


def test_load_profiles_returns_known_defaults():
    profiles = load_profiles(ROOT_PROFILES)

    assert isinstance(profiles, ProfileSet)
    assert profiles.default == "safe80"
    assert set(profiles.items.keys()) == {
        "safe80",
        "tp2",
        "tp2_2x32_smoke",
        "tp2_2x32_fp8_gpuonly",
    }

    safe80 = profiles.items["safe80"]
    assert isinstance(safe80, ModelProfile)
    assert safe80.target == "google/gemma-4-31B-it"
    assert safe80.drafter == "google/gemma-4-31B-it-assistant"
    assert safe80.num_speculative_tokens == 4
    assert safe80.tensor_parallel_size == 1
    assert safe80.gpu_memory_utilization == pytest.approx(0.90)
    assert safe80.max_model_len == 32768
    assert safe80.temperature == pytest.approx(0.0)
    assert safe80.top_p == pytest.approx(1.0)
    assert safe80.top_k == 0
    assert safe80.requires_vram_gb == 80
    assert safe80.cpu_offload_gb == pytest.approx(0.0)
    assert safe80.validation_level == "unverified"
    assert safe80.max_output_tokens == 4096

    smoke = profiles.items["tp2_2x32_smoke"]
    assert smoke.tensor_parallel_size == 2
    assert smoke.max_model_len == 2048
    assert smoke.cpu_offload_gb == pytest.approx(8.0)
    assert smoke.max_num_seqs == 1
    assert smoke.max_num_batched_tokens == 4096
    assert smoke.enforce_eager is True
    assert smoke.validation_level == "smoke"
    assert smoke.max_output_tokens == 1024

    gpu_only = profiles.items["tp2_2x32_fp8_gpuonly"]
    assert gpu_only.cpu_offload_gb == pytest.approx(0.0)
    assert gpu_only.quantization == "fp8"


def test_resolve_profile_via_alias():
    profiles = load_profiles(ROOT_PROFILES)
    profile = resolve_profile("gemma-4-31b-mtp", profiles)
    assert profile.name == "safe80"


def test_resolve_unknown_profile_raises():
    profiles = load_profiles(ROOT_PROFILES)
    with pytest.raises(KeyError):
        resolve_profile("nonexistent", profiles)


def test_invalid_num_speculative_tokens_rejected(tmp_path):
    bad = tmp_path / "profiles.yaml"
    bad.write_text(
        """default: bad
aliases: {}
profiles:
  bad:
    target: google/gemma-4-31B-it
    drafter: google/gemma-4-31B-it-assistant
    num_speculative_tokens: 0
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 4096
    temperature: 0.0
    top_p: 1.0
    top_k: 0
    requires_vram_gb: 80
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_profiles(bad)
