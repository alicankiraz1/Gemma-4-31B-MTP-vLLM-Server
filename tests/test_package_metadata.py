from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import gemma4_mtp_vllm


PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_version_matches_pyproject():
    data = _pyproject()
    assert gemma4_mtp_vllm.__version__ == data["project"]["version"]


def test_release_version_is_0_2_alpha_1():
    assert gemma4_mtp_vllm.__version__ == "0.2.0a1"


def test_required_vllm_min_version_present():
    assert gemma4_mtp_vllm.REQUIRED_VLLM_MIN_VERSION == "0.21.0"


def test_vllm_optional_extra_lists_min_version():
    data = _pyproject()
    extras = data["project"]["optional-dependencies"]
    vllm_entries = extras.get("vllm", [])
    assert "vllm==0.21.0" in vllm_entries
    assert "fastapi>=0.115,<0.137" in vllm_entries
    assert "prometheus-fastapi-instrumentator==7.1.0" in vllm_entries
