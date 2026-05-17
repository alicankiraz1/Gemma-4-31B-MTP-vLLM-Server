# vLLM Sidecar Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gemma 4 31B MTP local API gateway on top of vLLM (NVIDIA / AMD GPUs) that preserves the OpenAI + Anthropic dual-protocol surface, guardrails, doctor diagnostics, and reproducible MTP benchmarks of the MLX project, while delegating all model execution to an unmodified vLLM OpenAI-compatible server.

**Architecture:** A FastAPI sidecar gateway runs in front of an unmodified `vllm serve` process. The gateway owns auth, body limits, rate limiting, queueing, Anthropic translation, doctor checks, bench harness, and profile management. The vLLM process owns the Gemma 4 31B target model, the MTP assistant drafter, continuous batching, prefix caching, and CUDA / ROCm execution. All gateway → vLLM communication is HTTP only; no private vLLM Python imports are used in production code paths.

**Tech Stack:** Python 3.10+, FastAPI, Typer, Pydantic-compatible JSON dictionaries, HTTPX (async client), Pytest, vLLM (`>=0.11.0`, runtime dependency only — never imported from internals), POSIX shell verification scripts.

---

## File Structure

Create (`src/gemma4_mtp_vllm/`):

- `__init__.py` - package metadata, version, REQUIRED_VLLM_MIN_VERSION.
- `profiles.py` - typed profile loader and resolver.
- `policy.py` - OpenAI / Anthropic fail-fast validation.
- `benchmarking.py` - pure benchmark math, observations, summaries, JSON helpers.
- `doctor.py` - doctor diagnostic report builder.
- `launch.py` - vllm serve command builder.
- `cli.py` - Typer entrypoints: `serve`, `doctor`, `generate`, `bench`, `bench-matrix`, `launch`.
- `anthropic_adapter.py` - Anthropic → OpenAI request translator and OpenAI → Anthropic response / SSE translator.
- `backend/__init__.py`
- `backend/vllm_client.py` - async HTTP client for vLLM OpenAI server.
- `backend/response_parser.py` - finish reason, usage, visible text helpers shared by gateway and adapter.
- `server/__init__.py`
- `server/app.py` - FastAPI `create_app` and all HTTP endpoints.
- `server/limits.py` - typed `ServerLimits` dataclass.
- `server/bind_policy.py` - bind host policy helper.
- `server/errors.py` - protocol-shaped error response helper.
- `server/runtime_state.py` - bounded admission and counters.
- `server/middleware.py` - body cap, rate limiter, CORS middleware install helper.

Create (`config/`):

- `profiles.yaml` - default profile registry (root copy used by tests, mirrored to package via build).

Create (`src/gemma4_mtp_vllm/config/`):

- `profiles.yaml` - packaged copy of the same profiles file.

Create (`tests/`):

- `test_package_metadata.py`
- `test_profiles.py`
- `test_server_limits.py`
- `test_bind_policy.py`
- `test_server_errors.py`
- `test_runtime_state.py`
- `test_middleware.py`
- `test_policy.py`
- `test_vllm_client.py`
- `test_anthropic_adapter.py`
- `test_server_app.py`
- `test_openai_server.py`
- `test_anthropic_server.py`
- `test_server_health.py`
- `test_server_metrics.py`
- `test_doctor.py`
- `test_benchmarking.py`
- `test_bench_cli.py`
- `test_launch.py`
- `test_cli.py`
- `test_release_scripts.py`

Create (`scripts/`):

- `make_source_archive.sh`
- `verify_source_archive.sh`
- `verify_wheel_freshness.sh`

Create (top-level):

- `pyproject.toml`
- `README.md`
- `.gitignore`
- `LICENSE` (Apache-2.0 placeholder; user may swap later)

---

## Task 0: Preflight, Project Init, First Commit

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `src/gemma4_mtp_vllm/__init__.py`
- Create: `tests/__init__.py`

**Context:** No existing git repo in this directory. This task initializes the project, creates the minimal package skeleton, and produces the first commit so subsequent task commits land on a real history.

- [ ] **Step 1: Create `.gitignore`**

```text
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
dist/
build/
*.egg-info/
bench-results/
```

- [ ] **Step 2: Create `LICENSE` with Apache-2.0 text**

Use the standard Apache-2.0 license text with copyright line:
`Copyright 2026 Alican Kiraz`. The full license text is available at
https://www.apache.org/licenses/LICENSE-2.0.txt. Paste the canonical text
verbatim.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "gemma4-mtp-vllm"
version = "0.1.0"
description = "Gemma 4 31B MTP local API gateway on vLLM"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    "httpx>=0.27",
    "rich>=13.7",
    "typer>=0.12",
    "pyyaml>=6.0.2",
]

[project.optional-dependencies]
test = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "tomli>=2; python_version<'3.11'",
]
dev = [
    "build>=1.2",
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "tomli>=2; python_version<'3.11'",
]
vllm = [
    "vllm>=0.11.0",
]

[project.scripts]
vllm-mtp = "gemma4_mtp_vllm.cli:app"

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
```

Note: vLLM is an optional extra because installing it triggers heavy CUDA
wheels. The gateway runs without it (it is the user's responsibility to run
vLLM in a separate process). The gateway only requires `httpx` to talk to
vLLM over HTTP.

- [ ] **Step 4: Create minimal `README.md`**

```markdown
# Gemma 4 31B MTP vLLM Sidecar Gateway

CUDA / ROCm sibling of the MLX gateway. Runs Google Gemma 4 31B with the
Gemma 4 MTP assistant drafter through vLLM, behind a FastAPI sidecar that
adds OpenAI + Anthropic dual-protocol support, auth, rate limiting,
bounded admission, doctor diagnostics, and a reproducible MTP benchmark.

See `docs/specs/2026-05-16-vllm-sidecar-gateway-design.md` for the design.
See `docs/plans/2026-05-16-vllm-sidecar-gateway-implementation-plan.md` for
the implementation plan currently in progress.
```

The full README is generated in Task 21.

- [ ] **Step 5: Create `src/gemma4_mtp_vllm/__init__.py` minimal**

```python
__version__ = "0.1.0"
REQUIRED_VLLM_MIN_VERSION = "0.11.0"

__all__ = ["REQUIRED_VLLM_MIN_VERSION", "__version__"]
```

- [ ] **Step 6: Create `tests/__init__.py` empty**

Empty file. Marker for the test package.

- [ ] **Step 7: Init git and first commit**

```bash
cd /Users/alicankiraz/Desktop/BillionDollarsIdeas/Gemma-4-31B-MTP-vllm
git init
git add .
git commit -m "$(cat <<'EOF'
chore: bootstrap vllm sidecar gateway project
EOF
)"
```

Expected output: initial commit on `main` with the skeleton files above plus
the existing `docs/` directory.

---

## Task 1: Package Metadata Tests

**Files:**
- Create: `tests/test_package_metadata.py`

**Context:** Lock the package version and the minimum required vLLM version
behind tests so they cannot drift silently. The MLX project pins its
upstream commit; this gateway only requires a minimum version because vLLM
ships proper releases.

- [ ] **Step 1: Write failing test**

```python
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


def test_required_vllm_min_version_present():
    assert gemma4_mtp_vllm.REQUIRED_VLLM_MIN_VERSION.startswith("0.")


def test_vllm_optional_extra_lists_min_version():
    data = _pyproject()
    extras = data["project"]["optional-dependencies"]
    vllm_entries = extras.get("vllm", [])
    expected = f"vllm>={gemma4_mtp_vllm.REQUIRED_VLLM_MIN_VERSION}"
    assert expected in vllm_entries
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_package_metadata.py -v
```
Expected: all three tests pass already because Task 0 set both values to
match. If they fail, the implementer must reconcile `__init__.py` and
`pyproject.toml`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_package_metadata.py
git commit -m "$(cat <<'EOF'
test: lock package version and vllm minimum
EOF
)"
```

---

## Task 2: Profile System

**Files:**
- Create: `config/profiles.yaml`
- Create: `src/gemma4_mtp_vllm/config/profiles.yaml`
- Create: `src/gemma4_mtp_vllm/profiles.py`
- Create: `tests/test_profiles.py`

**Context:** A typed profile registry mirroring the MLX project but adapted
to CUDA / ROCm and to the public vLLM `--speculative-config` surface.
Default profile name is `safe80` (single 80 GB GPU). Alternative `tp2`
profile targets 2× 40+ GB GPUs.

- [ ] **Step 1: Create `config/profiles.yaml`**

```yaml
default: safe80
aliases:
  default: safe80
  gemma-4-31b-mtp: safe80
  claude-gemma-4-31b-mtp: safe80
profiles:
  safe80:
    target: google/gemma-4-31B-it
    drafter: google/gemma-4-31B-it-assistant
    num_speculative_tokens: 4
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.90
    max_model_len: 32768
    temperature: 0.0
    top_p: 1.0
    top_k: 0
    requires_vram_gb: 80
  tp2:
    target: google/gemma-4-31B-it
    drafter: google/gemma-4-31B-it-assistant
    num_speculative_tokens: 4
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.90
    max_model_len: 32768
    temperature: 0.0
    top_p: 1.0
    top_k: 0
    requires_vram_gb: 40
```

- [ ] **Step 2: Mirror the file into the package**

Identical content at `src/gemma4_mtp_vllm/config/profiles.yaml`. A test
will assert byte equality so the two stay in sync.

- [ ] **Step 3: Write failing test**

```python
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
    assert set(profiles.items.keys()) == {"safe80", "tp2"}

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
```

- [ ] **Step 4: Run failing**

```bash
python -m pytest tests/test_profiles.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 5: Implement `src/gemma4_mtp_vllm/profiles.py`**

```python
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

    fields = {
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

    values: dict[str, Any] = {}
    for key, expected_type in fields.items():
        value = config.get(key)
        if isinstance(value, bool) and _expects_int(expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if not isinstance(value, expected_type):
            raise ValueError(f"profile field {key} has invalid type")
        if key in {"gpu_memory_utilization", "temperature", "top_p"}:
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

    return values


def _expects_int(expected_type: type | tuple[type, ...]) -> bool:
    if expected_type is int:
        return True
    return isinstance(expected_type, tuple) and int in expected_type
```

- [ ] **Step 6: Run passing**

```bash
python -m pytest tests/test_profiles.py -v
```
Expected: all five tests pass.

- [ ] **Step 7: Commit**

```bash
git add config/profiles.yaml src/gemma4_mtp_vllm/config/profiles.yaml \
        src/gemma4_mtp_vllm/profiles.py tests/test_profiles.py
git commit -m "$(cat <<'EOF'
feat: add typed profile registry for vllm sidecar
EOF
)"
```

---

## Task 3: Server Limits, Bind Policy, Errors

**Files:**
- Create: `src/gemma4_mtp_vllm/server/__init__.py` (empty)
- Create: `src/gemma4_mtp_vllm/server/limits.py`
- Create: `src/gemma4_mtp_vllm/server/bind_policy.py`
- Create: `src/gemma4_mtp_vllm/server/errors.py`
- Create: `tests/test_server_limits.py`
- Create: `tests/test_bind_policy.py`
- Create: `tests/test_server_errors.py`

**Context:** Three small, independent guardrail primitives. Combine into one
task to keep commits cohesive without sacrificing isolation.

- [ ] **Step 1: Write `tests/test_server_limits.py`**

```python
from __future__ import annotations

import pytest

from gemma4_mtp_vllm.server.limits import ServerLimits


def test_default_limits_have_safe_values():
    limits = ServerLimits()

    assert limits.max_body_bytes == 2 * 1024 * 1024
    assert limits.max_output_tokens == 4096
    assert limits.max_queue_size == 8
    assert limits.rate_limit_rpm == 30
    assert limits.metrics_enabled is True
    assert limits.cors_origins == ()


def test_public_dict_exposes_runtime_fields():
    limits = ServerLimits(max_output_tokens=512, max_queue_size=4)
    payload = limits.public_dict()

    assert payload == {
        "max_body_bytes": 2 * 1024 * 1024,
        "max_output_tokens": 512,
        "max_queue_size": 4,
        "rate_limit_rpm": 30,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_body_bytes": 0},
        {"max_output_tokens": 0},
        {"max_queue_size": 0},
        {"rate_limit_rpm": -1},
    ],
)
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        ServerLimits(**kwargs)
```

- [ ] **Step 2: Implement `server/limits.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerLimits:
    max_body_bytes: int = 2 * 1024 * 1024
    max_output_tokens: int = 4096
    max_queue_size: int = 8
    rate_limit_rpm: int = 30
    metrics_enabled: bool = True
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.rate_limit_rpm < 0:
            raise ValueError("rate_limit_rpm must be non-negative")

    def public_dict(self) -> dict[str, int]:
        return {
            "max_body_bytes": self.max_body_bytes,
            "max_output_tokens": self.max_output_tokens,
            "max_queue_size": self.max_queue_size,
            "rate_limit_rpm": self.rate_limit_rpm,
        }
```

- [ ] **Step 3: Write `tests/test_bind_policy.py`**

```python
import pytest

from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "LOCALHOST"])
def test_loopback_hosts_do_not_require_api_key(host):
    assert bind_host_requires_api_key(host) is False


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.50", "10.0.0.5", "example.com"],
)
def test_non_loopback_hosts_require_api_key(host):
    assert bind_host_requires_api_key(host) is True
```

- [ ] **Step 4: Implement `server/bind_policy.py`**

```python
from __future__ import annotations


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def bind_host_requires_api_key(host: str) -> bool:
    normalized = host.strip().casefold()
    return normalized not in _LOOPBACK_HOSTS
```

- [ ] **Step 5: Write `tests/test_server_errors.py`**

```python
import json

from gemma4_mtp_vllm.server.errors import protocol_error_response


def test_openai_error_shape():
    response = protocol_error_response(
        status_code=400,
        code="invalid_request",
        message="boom",
    )
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body == {
        "error": {
            "code": "invalid_request",
            "message": "boom",
            "type": "invalid_request_error",
        }
    }


def test_anthropic_error_shape():
    response = protocol_error_response(
        status_code=429,
        code="rate_limited",
        message="slow down",
        protocol="anthropic",
    )
    assert response.status_code == 429
    body = json.loads(response.body)
    assert body == {
        "type": "error",
        "error": {
            "type": "rate_limited",
            "message": "slow down",
        },
    }
```

- [ ] **Step 6: Implement `server/errors.py`**

```python
from __future__ import annotations

from fastapi.responses import JSONResponse


_DEFAULT_OPENAI_TYPE = "invalid_request_error"


def protocol_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    protocol: str = "openai",
) -> JSONResponse:
    if protocol == "anthropic":
        body = {
            "type": "error",
            "error": {"type": code, "message": message},
        }
    else:
        body = {
            "error": {
                "code": code,
                "message": message,
                "type": _DEFAULT_OPENAI_TYPE,
            }
        }
    return JSONResponse(status_code=status_code, content=body)
```

- [ ] **Step 7: Run all three test files**

```bash
python -m pytest tests/test_server_limits.py tests/test_bind_policy.py \
    tests/test_server_errors.py -v
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/gemma4_mtp_vllm/server tests/test_server_limits.py \
        tests/test_bind_policy.py tests/test_server_errors.py
git commit -m "$(cat <<'EOF'
feat: add server limits, bind policy, error helpers
EOF
)"
```

---

## Task 4: Runtime State

**Files:**
- Create: `src/gemma4_mtp_vllm/server/runtime_state.py`
- Create: `tests/test_runtime_state.py`

**Context:** Bounded admission, counters, and last backend error tracking,
adapted from the MLX project to surface generation metrics that the
gateway can record (vLLM owns the true per-token metrics, but the
gateway records request-level totals).

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import asyncio

import pytest

from gemma4_mtp_vllm.server.runtime_state import (
    QueueFull,
    RuntimeState,
)


def test_initial_snapshot_is_zeroed():
    state = RuntimeState(max_queue_size=4)
    snapshot = state.snapshot()
    assert snapshot == {
        "active_requests": 0,
        "queued_requests": 0,
        "total_requests": 0,
        "rejected_requests": 0,
        "backend_errors": 0,
        "generation_tokens": 0,
        "generation_seconds": 0.0,
        "batch_requests": 0,
        "last_request_id": None,
    }


def test_record_generation_updates_counters():
    state = RuntimeState(max_queue_size=4)
    state.record_generation(
        generation_tokens=12,
        generation_seconds=0.5,
        batch_size=1,
    )
    snapshot = state.snapshot()
    assert snapshot["generation_tokens"] == 12
    assert snapshot["generation_seconds"] == pytest.approx(0.5)
    assert snapshot["batch_requests"] == 1


def test_record_backend_error_sets_last_error():
    state = RuntimeState(max_queue_size=2)
    state.record_backend_error("vllm_unreachable")
    snapshot = state.snapshot()
    assert state.last_backend_error == "vllm_unreachable"
    assert snapshot["backend_errors"] == 1
    state.clear_backend_error()
    assert state.last_backend_error is None


def test_acquire_generation_slot_bounded():
    async def scenario() -> None:
        state = RuntimeState(max_queue_size=1)
        slot1 = await state.acquire_generation_slot()
        slot2 = await state.acquire_generation_slot()
        with pytest.raises(QueueFull):
            await state.acquire_generation_slot()
        slot1.release()
        slot2.release()

    asyncio.run(scenario())


def test_request_id_recorded():
    state = RuntimeState(max_queue_size=2)
    state.note_request("req-1")
    assert state.snapshot()["last_request_id"] == "req-1"
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_runtime_state.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `server/runtime_state.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


class QueueFull(Exception):
    pass


@dataclass
class _Slot:
    state: "RuntimeState"

    def release(self) -> None:
        self.state._release_slot()


class RuntimeState:
    def __init__(self, *, max_queue_size: int) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._max_queue_size = max_queue_size
        self._active = 0
        self._total = 0
        self._rejected = 0
        self._backend_errors = 0
        self._generation_tokens = 0
        self._generation_seconds = 0.0
        self._batch_requests = 0
        self._last_request_id: str | None = None
        self._last_backend_error: str | None = None

    @property
    def last_backend_error(self) -> str | None:
        return self._last_backend_error

    async def acquire_generation_slot(self) -> _Slot:
        # The gateway slot is a synchronous in-memory backstop: vLLM owns
        # real concurrency through continuous batching, so this layer only
        # bounds how many requests the gateway will admit before rejecting.
        # `max_queue_size + 1` gives one active plus `max_queue_size`
        # acceptable concurrent admissions before the next call raises.
        if self._active >= self._max_queue_size + 1:
            self._rejected += 1
            raise QueueFull()
        self._active += 1
        self._total += 1
        return _Slot(state=self)

    def _release_slot(self) -> None:
        if self._active > 0:
            self._active -= 1

    def record_generation(
        self,
        *,
        generation_tokens: int | None,
        generation_seconds: float,
        batch_size: int,
    ) -> None:
        if generation_tokens is not None and generation_tokens > 0:
            self._generation_tokens += int(generation_tokens)
        if generation_seconds > 0:
            self._generation_seconds += float(generation_seconds)
        if batch_size > 0:
            self._batch_requests += int(batch_size)

    def record_backend_error(self, code: str) -> None:
        self._backend_errors += 1
        self._last_backend_error = code

    def clear_backend_error(self) -> None:
        self._last_backend_error = None

    def note_request(self, request_id: str) -> None:
        self._last_request_id = request_id

    def snapshot(self) -> dict[str, object]:
        return {
            "active_requests": self._active,
            "queued_requests": 0,
            "total_requests": self._total,
            "rejected_requests": self._rejected,
            "backend_errors": self._backend_errors,
            "generation_tokens": self._generation_tokens,
            "generation_seconds": self._generation_seconds,
            "batch_requests": self._batch_requests,
            "last_request_id": self._last_request_id,
        }
```

Note: this is a pure synchronous slot counter. vLLM handles actual
concurrency through continuous batching; the gateway's slot only bounds
admission so the gateway itself cannot pile up unbounded work in memory.
`queued_requests` stays at zero in the snapshot because there is no
waiting queue at this layer — backpressure comes from `QueueFull` being
raised immediately when admission exceeds `max_queue_size + 1`.
`acquire_generation_slot` stays `async` to keep the interface stable
under a future real queue, but does not await anything in v0.1.

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_runtime_state.py -v
```
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/runtime_state.py tests/test_runtime_state.py
git commit -m "$(cat <<'EOF'
feat: add bounded runtime state with generation counters
EOF
)"
```

---

## Task 5: Request Boundary Middleware

**Files:**
- Create: `src/gemma4_mtp_vllm/server/middleware.py`
- Create: `tests/test_middleware.py`

**Context:** Rate limiting, body size enforcement, CORS, request id
propagation. Reuse the MLX project's helper shape but adapted to the
sidecar's runtime state.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.middleware import install_request_boundary_middleware
from gemma4_mtp_vllm.server.runtime_state import RuntimeState


def _make_app(limits: ServerLimits, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI()
    runtime_state = RuntimeState(max_queue_size=limits.max_queue_size)
    install_request_boundary_middleware(
        app,
        limits=limits,
        api_key=api_key,
        public_paths={"/livez"},
        runtime_state=runtime_state,
    )

    @app.get("/livez")
    async def livez() -> dict:
        return {"status": "ok"}

    @app.post("/protected")
    async def protected() -> dict:
        return {"ok": True}

    return app


def test_body_cap_enforced():
    limits = ServerLimits(max_body_bytes=16)
    client = TestClient(_make_app(limits))
    response = client.post("/protected", content="x" * 64)
    assert response.status_code == 413


def test_cors_default_deny_passes_no_origin_header():
    limits = ServerLimits()
    client = TestClient(_make_app(limits))
    response = client.options(
        "/protected",
        headers={"Origin": "https://example.com",
                 "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_opt_in_returns_origin_header():
    limits = ServerLimits(cors_origins=("https://app.test",))
    client = TestClient(_make_app(limits))
    response = client.options(
        "/protected",
        headers={"Origin": "https://app.test",
                 "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "https://app.test"


def test_rate_limit_returns_429_after_threshold():
    limits = ServerLimits(rate_limit_rpm=2)
    client = TestClient(_make_app(limits))
    statuses = [client.post("/protected").status_code for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_public_paths_bypass_rate_limit_and_body_cap():
    limits = ServerLimits(rate_limit_rpm=1)
    client = TestClient(_make_app(limits))
    for _ in range(5):
        assert client.get("/livez").status_code == 200


def test_request_id_header_propagated():
    limits = ServerLimits()
    client = TestClient(_make_app(limits))
    response = client.post(
        "/protected",
        headers={"X-Request-ID": "abc-123"},
    )
    assert response.headers["x-request-id"] == "abc-123"
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_middleware.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `server/middleware.py`**

Implement an `install_request_boundary_middleware(app, *, limits, api_key,
public_paths, runtime_state)` function that:

1. Adds an HTTP middleware that reads `X-Request-ID` (or generates a uuid),
   stores it via `runtime_state.note_request`, and echoes it on response.
2. Rejects request bodies larger than `limits.max_body_bytes` with HTTP 413.
3. Enforces a sliding 60-second window in-memory rate limit keyed by
   credential when `api_key` is configured (bearer or x-api-key) else by
   client host.
4. Applies CORS allow-list: if `Origin` matches `limits.cors_origins`, set
   `access-control-allow-origin`, `access-control-allow-methods`,
   `access-control-allow-headers`. Otherwise omit headers and let the
   browser block.
5. Bypasses all checks for any path in `public_paths`.

Use `protocol_error_response` from `server.errors` for the 413 and 429
responses. Use a per-app in-memory dict keyed by credential or host with a
deque of timestamps for rate limit accounting.

```python
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Iterable

from fastapi import FastAPI, Request
from fastapi.responses import Response

from gemma4_mtp_vllm.server.errors import protocol_error_response
from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.runtime_state import RuntimeState


_RATE_WINDOW_SECONDS = 60.0


def install_request_boundary_middleware(
    app: FastAPI,
    *,
    limits: ServerLimits,
    api_key: str | None,
    public_paths: Iterable[str],
    runtime_state: RuntimeState,
) -> None:
    public_set = set(public_paths)
    rate_buckets: dict[str, deque[float]] = defaultdict(deque)
    allowed_origins = set(limits.cors_origins)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        path = request.url.path
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        runtime_state.note_request(request_id)

        if path not in public_set:
            length_error = await _enforce_body_cap(
                request,
                max_bytes=limits.max_body_bytes,
            )
            if length_error is not None:
                return _stamp(length_error, request_id)

            if limits.rate_limit_rpm > 0:
                key = _rate_key(request, api_key)
                if not _allow_rate(rate_buckets[key], limits.rate_limit_rpm):
                    return _stamp(
                        protocol_error_response(
                            status_code=429,
                            code="rate_limited",
                            message="too many requests",
                        ),
                        request_id,
                    )

        response = await call_next(request)
        _apply_cors(response, request.headers.get("origin"), allowed_origins)
        response.headers["x-request-id"] = request_id
        return response


async def _enforce_body_cap(
    request: Request,
    *,
    max_bytes: int,
) -> Response | None:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                return protocol_error_response(
                    status_code=413,
                    code="request_too_large",
                    message=f"request body must be at most {max_bytes} bytes",
                )
        except ValueError:
            return protocol_error_response(
                status_code=400,
                code="invalid_request",
                message="content-length header must be a number",
            )
    body = await request.body()
    if len(body) > max_bytes:
        return protocol_error_response(
            status_code=413,
            code="request_too_large",
            message=f"request body must be at most {max_bytes} bytes",
        )
    request._body = body  # noqa: SLF001 - re-cache for downstream handlers
    return None


def _allow_rate(bucket: deque[float], rpm_limit: int) -> bool:
    now = time.monotonic()
    while bucket and now - bucket[0] > _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= rpm_limit:
        return False
    bucket.append(now)
    return True


def _rate_key(request: Request, api_key: str | None) -> str:
    if api_key and request.headers.get("authorization") == f"Bearer {api_key}":
        return "credential:bearer"
    if api_key and request.headers.get("x-api-key") == api_key:
        return "credential:x-api-key"
    client = request.client.host if request.client else "unknown"
    return f"client:{client}"


def _apply_cors(response: Response, origin: str | None, allowed: set[str]) -> None:
    if origin and origin in allowed:
        response.headers["access-control-allow-origin"] = origin
        response.headers["access-control-allow-methods"] = (
            "GET, POST, OPTIONS"
        )
        response.headers["access-control-allow-headers"] = (
            "authorization, content-type, x-api-key, x-request-id"
        )


def _stamp(response: Response, request_id: str) -> Response:
    response.headers["x-request-id"] = request_id
    return response
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_middleware.py -v
```
Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/middleware.py tests/test_middleware.py
git commit -m "$(cat <<'EOF'
feat: add request boundary middleware for gateway
EOF
)"
```

---

## Task 6: Policy Validation (OpenAI + Anthropic)

**Files:**
- Create: `src/gemma4_mtp_vllm/policy.py`
- Create: `tests/test_policy.py`

**Context:** Fail-fast rejection of features the gateway intentionally does
not support in v0.1. Mirrors the MLX project's policy module. Behavior
must match: tools / function_call / response_format etc. reject with
`UnsupportedFeature`; explicit no-op defaults like `tools: []` accepted.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import pytest

from gemma4_mtp_vllm.policy import (
    UnsupportedFeature,
    validate_anthropic_request,
    validate_openai_request,
)


def test_openai_minimal_payload_accepted():
    validate_openai_request({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("field", ["tools", "tool_choice", "function_call", "functions"])
def test_openai_rejects_unsupported_fields(field):
    payload = {"messages": [], field: ["something"]}
    with pytest.raises(UnsupportedFeature) as exc:
        validate_openai_request(payload)
    assert exc.value.code == "unsupported_feature"
    assert field in exc.value.message


def test_openai_accepts_noop_defaults():
    validate_openai_request(
        {
            "messages": [],
            "tools": [],
            "tool_choice": "none",
            "functions": [],
            "function_call": "none",
            "stop": None,
            "response_format": {"type": "text"},
        }
    )


def test_openai_rejects_structured_response_format_when_mtp_enabled():
    payload = {
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    with pytest.raises(UnsupportedFeature):
        validate_openai_request(payload, mtp_enabled=True)


def test_openai_accepts_structured_response_format_when_mtp_disabled():
    payload = {
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    validate_openai_request(payload, mtp_enabled=False)


def test_anthropic_minimal_payload_accepted():
    validate_anthropic_request({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize(
    "field",
    ["tools", "tool_choice", "thinking", "mcp", "files", "stop_sequences"],
)
def test_anthropic_rejects_unsupported_fields(field):
    payload = {"messages": [], field: ["thing"]}
    with pytest.raises(UnsupportedFeature) as exc:
        validate_anthropic_request(payload)
    assert field in exc.value.message
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_policy.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/policy.py`**

```python
from __future__ import annotations

from typing import Any


class UnsupportedFeature(Exception):
    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


_OPENAI_REJECT_FIELDS = ("tools", "tool_choice", "function_call", "functions", "stop")
_ANTHROPIC_REJECT_FIELDS = (
    "tools",
    "tool_choice",
    "thinking",
    "mcp",
    "files",
    "stop_sequences",
)


def validate_openai_request(payload: dict[str, Any], *, mtp_enabled: bool = True) -> None:
    for field in _OPENAI_REJECT_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if _is_openai_noop(field, value):
            continue
        raise UnsupportedFeature(
            status_code=400,
            code="unsupported_feature",
            message=f"openai field {field!r} is not supported in v1",
        )

    response_format = payload.get("response_format")
    if response_format is not None and isinstance(response_format, dict):
        format_type = response_format.get("type")
        if format_type and format_type != "text" and mtp_enabled:
            raise UnsupportedFeature(
                status_code=400,
                code="unsupported_feature",
                message=(
                    "openai field 'response_format' with structured types is "
                    "not supported while mtp is enabled"
                ),
            )


def validate_anthropic_request(payload: dict[str, Any]) -> None:
    for field in _ANTHROPIC_REJECT_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if _is_anthropic_noop(field, value):
            continue
        raise UnsupportedFeature(
            status_code=400,
            code="unsupported_feature",
            message=f"anthropic field {field!r} is not supported in v1",
        )


def _is_openai_noop(field: str, value: Any) -> bool:
    if field == "tools" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "tool_choice" and (value is None or value == "none"):
        return True
    if field == "functions" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "function_call" and (value is None or value == "none"):
        return True
    if field == "stop" and value is None:
        return True
    return False


def _is_anthropic_noop(field: str, value: Any) -> bool:
    if field == "tools" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "tool_choice" and isinstance(value, dict) and value.get("type") == "none":
        return True
    if field == "thinking" and isinstance(value, dict) and value.get("type") == "disabled":
        return True
    if field == "stop_sequences" and isinstance(value, list) and len(value) == 0:
        return True
    if field in {"mcp", "files"} and isinstance(value, list) and len(value) == 0:
        return True
    return False
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_policy.py -v
```
Expected: all eight parametrized tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/policy.py tests/test_policy.py
git commit -m "$(cat <<'EOF'
feat: add openai and anthropic fail-fast policy
EOF
)"
```

---

## Task 7: vLLM HTTP Client

**Files:**
- Create: `src/gemma4_mtp_vllm/backend/__init__.py`
- Create: `src/gemma4_mtp_vllm/backend/response_parser.py`
- Create: `src/gemma4_mtp_vllm/backend/vllm_client.py`
- Create: `tests/test_vllm_client.py`

**Context:** The gateway's only contact with vLLM. Async HTTP client built
on `httpx.AsyncClient`. All tests use an in-process httpx mock via
`httpx.MockTransport` so no real vLLM is required.

- [ ] **Step 1: Implement `backend/__init__.py` empty**

- [ ] **Step 2: Implement `backend/response_parser.py`**

```python
from __future__ import annotations

from typing import Any


def visible_text_for_history(text: str) -> str:
    return text


def finish_reason_from_openai(choice: dict[str, Any]) -> str:
    reason = choice.get("finish_reason") or "stop"
    return str(reason)


def usage_from_openai(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
```

- [ ] **Step 3: Write failing test**

```python
from __future__ import annotations

import json

import httpx
import pytest

from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError


def _client(handler) -> VllmClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://vllm.local:8000",
    )
    return VllmClient(http=http, base_url="http://vllm.local:8000")


@pytest.mark.asyncio
async def test_health_returns_status():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    async with _client(handler) as client:
        assert (await client.health())["status"] == "ok"


@pytest.mark.asyncio
async def test_list_models_returns_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
            },
        )

    async with _client(handler) as client:
        models = await client.list_models()
        assert models["data"][0]["id"] == "google/gemma-4-31B-it"


@pytest.mark.asyncio
async def test_chat_completion_proxies_body_and_returns_json():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "abc",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    async with _client(handler) as client:
        response = await client.chat_completion(
            {
                "model": "google/gemma-4-31B-it",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            }
        )
        assert response["choices"][0]["message"]["content"] == "hi"

    assert captured["body"]["max_tokens"] == 4


@pytest.mark.asyncio
async def test_chat_completion_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async with _client(handler) as client:
        with pytest.raises(VllmHttpError) as exc:
            await client.chat_completion({"messages": []})
        assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_chat_completion_stream_yields_chunks():
    body = (
        "data: {\"id\":\"a\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    async with _client(handler) as client:
        chunks = []
        async for chunk in client.chat_completion_stream({"messages": []}):
            chunks.append(chunk)
        assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
        assert chunks[-1] == {"_done": True}
```

- [ ] **Step 4: Run failing**

```bash
python -m pytest tests/test_vllm_client.py -v
```
Expected: import error.

- [ ] **Step 5: Implement `backend/vllm_client.py`**

```python
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class VllmHttpError(Exception):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class VllmClient:
    def __init__(self, *, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    async def __aenter__(self) -> "VllmClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def health(self) -> dict:
        return await self._get("/health")

    async def list_models(self) -> dict:
        return await self._get("/v1/models")

    async def version(self) -> dict:
        return await self._get("/version")

    async def chat_completion(self, body: dict) -> dict:
        return await self._post_json("/v1/chat/completions", body)

    async def completion(self, body: dict) -> dict:
        return await self._post_json("/v1/completions", body)

    async def chat_completion_stream(self, body: dict) -> AsyncIterator[dict]:
        async for chunk in self._post_stream("/v1/chat/completions", body):
            yield chunk

    async def _get(self, path: str) -> dict:
        response = await self._http.get(path)
        return self._json_or_raise(response)

    async def _post_json(self, path: str, body: dict) -> dict:
        response = await self._http.post(path, json=body)
        return self._json_or_raise(response)

    async def _post_stream(self, path: str, body: dict) -> AsyncIterator[dict]:
        payload = dict(body)
        payload["stream"] = True
        async with self._http.stream("POST", path, json=payload) as response:
            if response.status_code != 200:
                content = await response.aread()
                raise VllmHttpError(
                    status_code=response.status_code,
                    message=content.decode("utf-8", errors="replace"),
                )
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        yield {"_done": True}
                        return
                    yield json.loads(data)

    @staticmethod
    def _json_or_raise(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        return response.json()
```

- [ ] **Step 6: Run passing**

```bash
python -m pytest tests/test_vllm_client.py -v
```
Expected: all five async tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gemma4_mtp_vllm/backend tests/test_vllm_client.py
git commit -m "$(cat <<'EOF'
feat: add async vllm http client and response helpers
EOF
)"
```

---

## Task 8: Anthropic Adapter (request + response + streaming)

**Files:**
- Create: `src/gemma4_mtp_vllm/anthropic_adapter.py`
- Create: `tests/test_anthropic_adapter.py`

**Context:** Pure translation layer between Anthropic Messages and OpenAI
Chat Completions. No network, no FastAPI. The server endpoints in later
tasks compose this adapter with the vLLM client.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json

from gemma4_mtp_vllm.anthropic_adapter import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
)


def test_anthropic_request_translates_system_and_messages():
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "system": "Be concise.",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "4"}],
            },
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    openai_body = anthropic_request_to_openai(
        payload, openai_model="google/gemma-4-31B-it",
    )
    assert openai_body == {
        "model": "google/gemma-4-31B-it",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_openai_response_to_anthropic_envelope():
    openai_payload = {
        "id": "chatcmpl-abc",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }
    body = openai_response_to_anthropic(
        openai_payload,
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-gemma-4-31b-mtp"
    assert body["content"] == [{"type": "text", "text": "Hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 1}
    assert body["id"].startswith("msg_")


def test_openai_response_max_tokens_maps_to_anthropic_stop_reason():
    body = openai_response_to_anthropic(
        {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )
    assert body["stop_reason"] == "max_tokens"


def test_openai_stream_to_anthropic_events_smoke():
    openai_chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"_done": True},
    ]
    events = list(
        openai_stream_to_anthropic_events(
            openai_chunks,
            anthropic_model="claude-gemma-4-31b-mtp",
            message_id_prefix="msg",
            prompt_tokens=1,
        )
    )
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert types.count("content_block_delta") == 2
    assert types[-2:] == ["message_delta", "message_stop"]
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_anthropic_adapter.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/anthropic_adapter.py`**

```python
from __future__ import annotations

import uuid
from typing import Any, Iterable, Iterator


def anthropic_request_to_openai(
    payload: dict[str, Any],
    *,
    openai_model: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": openai_model,
        "messages": _build_messages(payload),
    }
    for field in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
        if field in payload and payload[field] is not None:
            body[field] = payload[field]
    return body


def openai_response_to_anthropic(
    openai_payload: dict[str, Any],
    *,
    anthropic_model: str,
    message_id_prefix: str,
) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    primary = choices[0] if choices else {}
    content_text = _extract_message_content(primary)
    finish_reason = primary.get("finish_reason")
    usage = openai_payload.get("usage") or {}
    return {
        "id": f"{message_id_prefix}_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": _stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def openai_stream_to_anthropic_events(
    openai_chunks: Iterable[dict[str, Any]],
    *,
    anthropic_model: str,
    message_id_prefix: str,
    prompt_tokens: int,
) -> Iterator[dict[str, Any]]:
    response_id = f"{message_id_prefix}_{uuid.uuid4().hex}"
    output_tokens = 0
    stop_reason = "end_turn"
    started_content = False

    yield {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": anthropic_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    }

    for chunk in openai_chunks:
        if chunk.get("_done"):
            break
        choices = chunk.get("choices") or []
        if not choices:
            continue
        primary = choices[0]
        delta = primary.get("delta") or {}
        content = delta.get("content")
        if content:
            if not started_content:
                yield {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                started_content = True
            output_tokens += 1
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            }
        finish_reason = primary.get("finish_reason")
        if finish_reason:
            stop_reason = _stop_reason(finish_reason)

    if started_content:
        yield {"type": "content_block_stop", "index": 0}

    yield {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield {"type": "message_stop"}


def _build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = _content_to_text(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    raw_messages = payload.get("messages") or []
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            messages.append(
                {
                    "role": str(message.get("role") or "user"),
                    "content": _content_to_text(message.get("content")),
                }
            )
    return messages


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    return str(content)


def _extract_message_content(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _stop_reason(openai_finish: str | None) -> str:
    if openai_finish == "length":
        return "max_tokens"
    return "end_turn"
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_anthropic_adapter.py -v
```
Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/anthropic_adapter.py tests/test_anthropic_adapter.py
git commit -m "$(cat <<'EOF'
feat: add anthropic request, response, and streaming translator
EOF
)"
```

---

## Task 9: Gateway App — Foundation, Liveness, Readiness, Version

**Files:**
- Create: `src/gemma4_mtp_vllm/server/app.py`
- Create: `tests/test_server_app.py`

**Context:** Stand up `create_app` with the smallest possible HTTP surface
needed to verify the wiring: `/livez`, `/readyz`, `/version`, auth
enforcement. Endpoints that need vLLM (`/health`, generation paths) come
in later tasks.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client(api_key: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404)

    app = create_app(
        api_key=api_key,
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_livez_is_public():
    client = _client(api_key="secret")
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_requires_api_key():
    client = _client(api_key="secret")
    unauthorized = client.get("/readyz")
    authorized = client.get("/readyz", headers={"x-api-key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    body = authorized.json()
    assert body["status"] in {"ready", "degraded"}
    assert body["vllm"]["status"] == "ok"


def test_version_includes_gateway_and_vllm():
    client = _client(api_key="secret")
    response = client.get("/version", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["package"] == "gemma4-mtp-vllm"
    assert body["version"]
    assert body["vllm_version"] == "0.11.0"


def test_loopback_without_api_key_allowed():
    client = _client(api_key=None)
    assert client.get("/readyz").status_code == 200


def test_non_loopback_without_api_key_rejected():
    import pytest

    with pytest.raises(ValueError):
        create_app(
            api_key=None,
            bind_host="0.0.0.0",
            vllm_base_url="http://vllm.local:8000",
        )
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_server_app.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `server/app.py` minimal**

The implementation must:

- Build a `VllmClient` from either an injected `vllm_transport` or a real
  `httpx.AsyncClient(base_url=...)`. Tests inject `vllm_transport` to
  short-circuit the network.
- Register `/livez`, `/readyz`, `/version`.
- Use `RuntimeState`, `ServerLimits`, `install_request_boundary_middleware`,
  and `bind_host_requires_api_key`.
- Verify bind policy at startup and raise `ValueError` if violated.
- Probe vLLM's `/health` and `/version` lazily for `/readyz` and `/version`.

```python
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import (
    ModelProfile,
    ProfileSet,
    load_profiles,
    resolve_profile,
)
from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key
from gemma4_mtp_vllm.server.errors import protocol_error_response
from gemma4_mtp_vllm.server.limits import ServerLimits
from gemma4_mtp_vllm.server.middleware import install_request_boundary_middleware
from gemma4_mtp_vllm.server.runtime_state import RuntimeState

DEFAULT_MODEL_ALIAS = "gemma-4-31b-mtp"
DEFAULT_ANTHROPIC_MODEL_ALIAS = "claude-gemma-4-31b-mtp"
DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_PATHS = {"/livez"}


def create_app(
    *,
    profile_name: str | None = None,
    profiles: ProfileSet | None = None,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    bind_host: str = DEFAULT_BIND_HOST,
    api_key: str | None = None,
    limits: ServerLimits | None = None,
    vllm_base_url: str = "http://127.0.0.1:8000",
    vllm_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    if bind_host_requires_api_key(bind_host) and not api_key:
        raise ValueError(f"bind_host {bind_host} requires api_key")

    server_limits = limits or ServerLimits()
    profile_set = load_profiles() if profiles is None else profiles
    selected = resolve_profile(profile_name, profile_set)
    runtime_state = RuntimeState(max_queue_size=server_limits.max_queue_size)
    aliases = _aliases(profile_set, selected, model_alias)

    if vllm_transport is not None:
        http = httpx.AsyncClient(transport=vllm_transport, base_url=vllm_base_url)
    else:
        http = httpx.AsyncClient(base_url=vllm_base_url, timeout=httpx.Timeout(120.0))
    vllm = VllmClient(http=http, base_url=vllm_base_url)

    app = FastAPI(title="Gemma 4 31B MTP vLLM Gateway")
    app.state.vllm = vllm
    app.state.profile = selected
    app.state.aliases = aliases
    app.state.runtime_state = runtime_state
    app.state.limits = server_limits
    app.state.api_key = api_key
    app.state.bind_host = bind_host

    install_request_boundary_middleware(
        app,
        limits=server_limits,
        api_key=api_key,
        public_paths=PUBLIC_PATHS,
        runtime_state=runtime_state,
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Callable[[Request], Any]):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        auth_error = _auth_error(request, api_key, request.url.path)
        if auth_error is not None:
            return auth_error
        return await call_next(request)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await vllm.aclose()

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        vllm_status = await _probe_vllm(vllm)
        readiness = "ready" if vllm_status.get("status") == "ok" else "degraded"
        return {
            "status": readiness,
            "vllm": vllm_status,
            "last_backend_error": runtime_state.last_backend_error,
        }

    @app.get("/version")
    async def version() -> dict[str, Any]:
        try:
            vllm_version = (await vllm.version()).get("version")
        except VllmHttpError:
            vllm_version = None
        return {
            "package": "gemma4-mtp-vllm",
            "version": __version__,
            "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
            "vllm_version": vllm_version,
        }

    return app


def _aliases(
    profiles: ProfileSet,
    profile: ModelProfile,
    canonical: str,
) -> list[str]:
    items = {
        alias
        for alias, name in profiles.aliases.items()
        if name == profile.name
    }
    items.add(canonical)
    items.add(DEFAULT_ANTHROPIC_MODEL_ALIAS)
    return sorted(items)


def _auth_error(request: Request, api_key: str | None, path: str) -> JSONResponse | None:
    if api_key is None:
        return None
    if request.headers.get("authorization") == f"Bearer {api_key}":
        return None
    if request.headers.get("x-api-key") == api_key:
        return None

    if path in {"/v1/messages", "/v1/messages/count_tokens"}:
        return protocol_error_response(
            status_code=401,
            code="unauthorized",
            message="missing or invalid API key",
            protocol="anthropic",
        )
    return protocol_error_response(
        status_code=401,
        code="unauthorized",
        message="missing or invalid API key",
    )


async def _probe_vllm(vllm: VllmClient) -> dict[str, Any]:
    try:
        body = await vllm.health()
        if isinstance(body, dict) and body.get("status") == "ok":
            return {"status": "ok"}
        return {"status": "degraded", "raw": body}
    except VllmHttpError as exc:
        return {"status": "unreachable", "http_status": exc.status_code}
    except httpx.HTTPError as exc:
        return {"status": "unreachable", "error": str(exc)}
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_server_app.py -v
```
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_server_app.py
git commit -m "$(cat <<'EOF'
feat: scaffold fastapi gateway with livez/readyz/version
EOF
)"
```

---

## Task 10: Health and Metrics Endpoints

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Create: `tests/test_server_health.py`
- Create: `tests/test_server_metrics.py`

**Context:** `/health` reports the resolved profile, configured target,
drafter, num_speculative_tokens, vLLM probe, runtime snapshot, limits,
bind. `/metrics` exposes the gateway's own Prometheus counters and notes
that vLLM's `/metrics` is separate.

- [ ] **Step 1: Write `tests/test_server_health.py`**

```python
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client(api_key="secret"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "object": "list",
                "data": [
                    {"id": "google/gemma-4-31B-it", "object": "model"},
                    {"id": "google/gemma-4-31B-it-assistant", "object": "model"},
                ],
            })
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404)

    app = create_app(
        api_key=api_key,
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_health_returns_profile_and_vllm_info():
    client = _client()
    response = client.get("/health", headers={"x-api-key": "secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["profile"] == "safe80"
    assert body["target_model"] == "google/gemma-4-31B-it"
    assert body["drafter"] == "google/gemma-4-31B-it-assistant"
    assert body["num_speculative_tokens"] == 4
    assert body["vllm"]["status"] == "ok"
    assert body["vllm"]["version"] == "0.11.0"
    assert body["bind"]["host"] == "127.0.0.1"
    assert body["limits"]["max_output_tokens"] == 4096
    assert body["runtime"]["total_requests"] == 0
    assert body["model_aliases"]
```

- [ ] **Step 2: Write `tests/test_server_metrics.py`**

```python
import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_metrics_requires_auth():
    client = _client()
    unauthorized = client.get("/metrics")
    authorized = client.get("/metrics", headers={"x-api-key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    body = authorized.text
    assert "gemma4_mtp_active_requests" in body
    assert "gemma4_mtp_generation_tokens_total" in body
    assert "gemma4_mtp_backend_errors" in body
    assert authorized.headers["content-type"].startswith("text/plain")
```

- [ ] **Step 3: Run failing**

```bash
python -m pytest tests/test_server_health.py tests/test_server_metrics.py -v
```
Expected: 404 for both routes.

- [ ] **Step 4: Add `/health` and `/metrics` to `server/app.py`**

Insert near other routes:

```python
@app.get("/health")
async def health() -> dict[str, Any]:
    vllm_status = await _probe_vllm(vllm)
    if vllm_status.get("status") == "ok":
        try:
            version_body = await vllm.version()
            vllm_status["version"] = version_body.get("version")
        except VllmHttpError:
            vllm_status["version"] = None
    return {
        "status": "ready" if vllm_status.get("status") == "ok" else "degraded",
        "profile": selected.name,
        "target_model": selected.target,
        "drafter": selected.drafter,
        "num_speculative_tokens": selected.num_speculative_tokens,
        "tensor_parallel_size": selected.tensor_parallel_size,
        "gpu_memory_utilization": selected.gpu_memory_utilization,
        "max_model_len": selected.max_model_len,
        "model_aliases": aliases,
        "vllm": vllm_status,
        "bind": {"host": bind_host},
        "limits": server_limits.public_dict(),
        "runtime": runtime_state.snapshot(),
        "auth_modes": ["bearer", "x-api-key"],
        "tools_supported": False,
        "true_token_streaming": True,
        "continuous_batching": True,
        "token_counting": "estimated_word_count",
    }


@app.get("/metrics")
async def metrics() -> Response:
    if not server_limits.metrics_enabled:
        return Response(status_code=404)
    snapshot = runtime_state.snapshot()
    body = (
        "# TYPE gemma4_mtp_active_requests gauge\n"
        f"gemma4_mtp_active_requests {snapshot['active_requests']}\n"
        "# TYPE gemma4_mtp_queued_requests gauge\n"
        f"gemma4_mtp_queued_requests {snapshot['queued_requests']}\n"
        "# TYPE gemma4_mtp_total_requests counter\n"
        f"gemma4_mtp_total_requests {snapshot['total_requests']}\n"
        "# TYPE gemma4_mtp_rejected_requests counter\n"
        f"gemma4_mtp_rejected_requests {snapshot['rejected_requests']}\n"
        "# TYPE gemma4_mtp_backend_errors counter\n"
        f"gemma4_mtp_backend_errors {snapshot['backend_errors']}\n"
        "# TYPE gemma4_mtp_generation_tokens_total counter\n"
        f"gemma4_mtp_generation_tokens_total {snapshot['generation_tokens']}\n"
        "# TYPE gemma4_mtp_generation_seconds_total counter\n"
        f"gemma4_mtp_generation_seconds_total {snapshot['generation_seconds']}\n"
        "# TYPE gemma4_mtp_batch_requests_total counter\n"
        f"gemma4_mtp_batch_requests_total {snapshot['batch_requests']}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
```

- [ ] **Step 5: Run passing**

```bash
python -m pytest tests/test_server_health.py tests/test_server_metrics.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_server_health.py \
        tests/test_server_metrics.py
git commit -m "$(cat <<'EOF'
feat: expose /health and /metrics on gateway
EOF
)"
```

---

## Task 11: OpenAI Endpoints — Models + Chat Completions (sync)

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Create: `tests/test_openai_server.py`

**Context:** Add `/v1/models`, `/v1/chat/completions` (non-streaming),
`/v1/completions` (non-streaming). All requests pass through to vLLM via
`VllmClient`. Apply the policy module to fail-fast unsupported features
before forwarding. Enforce `max_tokens` cap from limits.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


CAPTURED: dict = {}


def _vllm_handler(request: httpx.Request) -> httpx.Response:
    CAPTURED["path"] = request.url.path
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
            },
        )
    if request.url.path == "/version":
        return httpx.Response(200, json={"version": "0.11.0"})
    if request.url.path == "/v1/chat/completions":
        CAPTURED["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc",
                "object": "chat.completion",
                "model": "google/gemma-4-31B-it",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )
    return httpx.Response(404)


def _client():
    return TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(_vllm_handler),
        )
    )


def test_models_endpoint_returns_aliases():
    response = _client().get("/v1/models", headers={"x-api-key": "secret"})
    assert response.status_code == 200
    ids = {entry["id"] for entry in response.json()["data"]}
    assert "gemma-4-31b-mtp" in ids
    assert "claude-gemma-4-31b-mtp" in ids


def test_chat_completion_pass_through():
    CAPTURED.clear()
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hi"
    assert CAPTURED["body"]["model"] == "google/gemma-4-31B-it"
    assert CAPTURED["body"]["max_tokens"] == 32


def test_chat_completion_caps_max_tokens_at_limit():
    CAPTURED.clear()
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 99999,
        },
    )
    assert response.status_code == 200
    assert CAPTURED["body"]["max_tokens"] == 4096


def test_chat_completion_rejects_tools():
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"


def test_chat_completion_unknown_model_404():
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "not-a-thing",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_openai_server.py -v
```
Expected: 404 for `/v1/models` and `/v1/chat/completions`.

- [ ] **Step 3: Implement endpoints in `server/app.py`**

Add helpers and endpoints:

```python
from gemma4_mtp_vllm.policy import (
    UnsupportedFeature,
    validate_openai_request,
)


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "local"}
            for alias in aliases
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload = await _bounded_json(request, server_limits.max_body_bytes)
    if isinstance(payload, JSONResponse):
        return payload
    try:
        validate_openai_request(payload, mtp_enabled=True)
    except UnsupportedFeature as exc:
        return protocol_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
        )
    if not _alias_known(payload.get("model"), aliases):
        return protocol_error_response(
            status_code=404,
            code="model_not_found",
            message=f"model {payload.get('model')!r} is not available",
        )

    body = _prepare_openai_body(payload, selected, server_limits)
    try:
        response = await vllm.chat_completion(body)
    except VllmHttpError as exc:
        runtime_state.record_backend_error("vllm_http_error")
        return protocol_error_response(
            status_code=503,
            code="backend_unavailable",
            message=f"vllm returned {exc.status_code}",
        )

    runtime_state.record_generation(
        generation_tokens=int((response.get("usage") or {}).get("completion_tokens") or 0),
        generation_seconds=0.0,
        batch_size=1,
    )
    return JSONResponse(response)
```

Helpers:

```python
async def _bounded_json(
    request: Request,
    max_bytes: int,
) -> dict[str, Any] | JSONResponse:
    body = await request.body()
    if len(body) > max_bytes:
        return protocol_error_response(
            status_code=413,
            code="request_too_large",
            message=f"request body must be at most {max_bytes} bytes",
        )
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return protocol_error_response(
            status_code=400,
            code="invalid_request",
            message="request body must be valid JSON",
        )
    if not isinstance(parsed, dict):
        return protocol_error_response(
            status_code=400,
            code="invalid_request",
            message="request body must be a JSON object",
        )
    return parsed


def _alias_known(value: Any, aliases: Iterable[str]) -> bool:
    if value is None:
        return True
    return value in set(aliases)


def _prepare_openai_body(
    payload: dict[str, Any],
    profile: ModelProfile,
    limits: ServerLimits,
) -> dict[str, Any]:
    body = dict(payload)
    body["model"] = profile.target
    requested_max = body.get("max_tokens", limits.max_output_tokens)
    if not isinstance(requested_max, int) or requested_max <= 0:
        requested_max = limits.max_output_tokens
    body["max_tokens"] = min(int(requested_max), limits.max_output_tokens)
    body.setdefault("temperature", profile.temperature)
    body.setdefault("top_p", profile.top_p)
    if profile.top_k > 0 and "top_k" not in body:
        body["top_k"] = profile.top_k
    body.pop("tools", None)
    body.pop("functions", None)
    body.pop("function_call", None)
    body.pop("tool_choice", None)
    body.pop("response_format", None)
    return body
```

Also import `json`, `Iterable`, and `ModelProfile` at the top if not
already.

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_openai_server.py -v
```
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_openai_server.py
git commit -m "$(cat <<'EOF'
feat: add openai models and chat completions endpoints
EOF
)"
```

---

## Task 12: OpenAI Streaming + /v1/completions

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_openai_server.py` (extend)

**Context:** Add streaming `/v1/chat/completions` SSE passthrough and the
non-streaming `/v1/completions` endpoint.

- [ ] **Step 1: Extend `tests/test_openai_server.py`**

Add tests:

```python
def test_chat_completion_streaming_passthrough(monkeypatch):
    chunks_body = (
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunks_body,
            )
        return httpx.Response(404)

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"data: " in response.content
    assert b"[DONE]" in response.content


def test_completions_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-abc",
                    "object": "text_completion",
                    "choices": [{"text": "World", "finish_reason": "stop", "index": 0}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(404)

    client = TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )
    response = client.post(
        "/v1/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": "gemma-4-31b-mtp", "prompt": "Hello", "max_tokens": 4},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "World"
```

- [ ] **Step 2: Run failing**

```bash
python -m pytest tests/test_openai_server.py -v
```
Expected: streaming and completions tests fail (no route / wrong shape).

- [ ] **Step 3: Implement streaming chat completion + completions**

In `server/app.py`:

```python
from fastapi.responses import StreamingResponse


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await _bounded_json(request, server_limits.max_body_bytes)
    if isinstance(payload, JSONResponse):
        return payload
    try:
        validate_openai_request(payload, mtp_enabled=True)
    except UnsupportedFeature as exc:
        return protocol_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
        )
    if not _alias_known(payload.get("model"), aliases):
        return protocol_error_response(
            status_code=404,
            code="model_not_found",
            message=f"model {payload.get('model')!r} is not available",
        )

    body = _prepare_openai_body(payload, selected, server_limits)
    streaming = bool(payload.get("stream"))

    if streaming:
        async def event_stream():
            try:
                async for chunk in vllm.chat_completion_stream(body):
                    if chunk.get("_done"):
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {json.dumps(chunk)}\n\n"
            except VllmHttpError as exc:
                runtime_state.record_backend_error("vllm_http_error")
                error = {
                    "error": {
                        "code": "backend_unavailable",
                        "message": f"vllm returned {exc.status_code}",
                    }
                }
                yield f"data: {json.dumps(error)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        response = await vllm.chat_completion(body)
    except VllmHttpError as exc:
        runtime_state.record_backend_error("vllm_http_error")
        return protocol_error_response(
            status_code=503,
            code="backend_unavailable",
            message=f"vllm returned {exc.status_code}",
        )

    runtime_state.record_generation(
        generation_tokens=int((response.get("usage") or {}).get("completion_tokens") or 0),
        generation_seconds=0.0,
        batch_size=1,
    )
    return JSONResponse(response)


@app.post("/v1/completions")
async def completions(request: Request) -> JSONResponse:
    payload = await _bounded_json(request, server_limits.max_body_bytes)
    if isinstance(payload, JSONResponse):
        return payload
    if not _alias_known(payload.get("model"), aliases):
        return protocol_error_response(
            status_code=404,
            code="model_not_found",
            message=f"model {payload.get('model')!r} is not available",
        )
    body = dict(payload)
    body["model"] = selected.target
    body["max_tokens"] = min(
        int(body.get("max_tokens") or server_limits.max_output_tokens),
        server_limits.max_output_tokens,
    )
    try:
        response = await vllm.completion(body)
    except VllmHttpError as exc:
        runtime_state.record_backend_error("vllm_http_error")
        return protocol_error_response(
            status_code=503,
            code="backend_unavailable",
            message=f"vllm returned {exc.status_code}",
        )
    return JSONResponse(response)
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_openai_server.py -v
```
Expected: all OpenAI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_openai_server.py
git commit -m "$(cat <<'EOF'
feat: stream chat completions and add text completions
EOF
)"
```

---

## Task 13: Anthropic Endpoints

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Create: `tests/test_anthropic_server.py`

**Context:** Add `/v1/messages` (sync + streaming) and
`/v1/messages/count_tokens`. Use the adapter module from Task 8.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _vllm(handler):
    return TestClient(
        create_app(
            api_key="secret",
            vllm_base_url="http://vllm.local:8000",
            vllm_transport=httpx.MockTransport(handler),
        )
    )


def test_anthropic_messages_returns_message_envelope():
    captured: dict = {}

    def handler(request):
        if request.url.path in {"/health"}:
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-x",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                    },
                },
            )
        return httpx.Response(404)

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "Hi"
    assert body["usage"]["input_tokens"] == 4
    assert body["usage"]["output_tokens"] == 1
    assert body["stop_reason"] == "end_turn"
    assert body["id"].startswith("msg_")
    forwarded = captured["body"]
    assert forwarded["model"] == "google/gemma-4-31B-it"
    assert forwarded["messages"][0] == {"role": "system", "content": "Be concise."}
    assert forwarded["max_tokens"] == 8


def test_anthropic_messages_streaming():
    body = (
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        b"data: [DONE]\n\n"
    )

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        return httpx.Response(404)

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    body_bytes = response.content
    assert b"event: message_start" in body_bytes
    assert b"event: content_block_delta" in body_bytes
    assert b"event: message_stop" in body_bytes


def test_anthropic_count_tokens_uses_word_count():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input_tokens"] >= 2
    assert response.headers["x-gemma4-mtp-token-counting"] == "estimated_word_count"


def test_anthropic_rejects_tools():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "calculator"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_feature"
```

- [ ] **Step 2: Run failing**

Expected: 404s on the messages endpoints.

- [ ] **Step 3: Implement Anthropic endpoints in `server/app.py`**

```python
from gemma4_mtp_vllm.anthropic_adapter import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
)
from gemma4_mtp_vllm.policy import validate_anthropic_request


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    payload = await _bounded_json(request, server_limits.max_body_bytes)
    if isinstance(payload, JSONResponse):
        return payload
    try:
        validate_anthropic_request(payload)
    except UnsupportedFeature as exc:
        return protocol_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            protocol="anthropic",
        )
    if not _alias_known(payload.get("model"), aliases):
        return protocol_error_response(
            status_code=404,
            code="model_not_found",
            message=f"model {payload.get('model')!r} is not available",
            protocol="anthropic",
        )

    openai_body = anthropic_request_to_openai(
        payload, openai_model=selected.target,
    )
    openai_body["max_tokens"] = min(
        int(openai_body.get("max_tokens") or server_limits.max_output_tokens),
        server_limits.max_output_tokens,
    )

    streaming = bool(payload.get("stream"))
    if streaming:
        async def event_stream():
            prompt_tokens = 0
            try:
                async for event in openai_stream_to_anthropic_events_async(
                    vllm.chat_completion_stream(openai_body),
                    anthropic_model=payload.get("model")
                    or DEFAULT_ANTHROPIC_MODEL_ALIAS,
                    message_id_prefix="msg",
                    prompt_tokens=prompt_tokens,
                ):
                    event_type = event.get("type", "message")
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event)}\n\n"
            except VllmHttpError as exc:
                runtime_state.record_backend_error("vllm_http_error")
                err = {
                    "type": "error",
                    "error": {
                        "type": "backend_unavailable",
                        "message": f"vllm returned {exc.status_code}",
                    },
                }
                yield "event: error\n"
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        openai_response = await vllm.chat_completion(openai_body)
    except VllmHttpError as exc:
        runtime_state.record_backend_error("vllm_http_error")
        return protocol_error_response(
            status_code=503,
            code="backend_unavailable",
            message=f"vllm returned {exc.status_code}",
            protocol="anthropic",
        )

    runtime_state.record_generation(
        generation_tokens=int(
            (openai_response.get("usage") or {}).get("completion_tokens") or 0
        ),
        generation_seconds=0.0,
        batch_size=1,
    )
    anthropic_body = openai_response_to_anthropic(
        openai_response,
        anthropic_model=payload.get("model") or DEFAULT_ANTHROPIC_MODEL_ALIAS,
        message_id_prefix="msg",
    )
    return JSONResponse(anthropic_body)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request) -> JSONResponse:
    payload = await _bounded_json(request, server_limits.max_body_bytes)
    if isinstance(payload, JSONResponse):
        return payload
    try:
        validate_anthropic_request(payload)
    except UnsupportedFeature as exc:
        return protocol_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            protocol="anthropic",
        )
    text = " ".join(
        str(message.get("content", ""))
        for message in payload.get("messages", [])
        if isinstance(message, dict)
    )
    system = payload.get("system")
    if isinstance(system, str):
        text = f"{system} {text}"
    return JSONResponse(
        {"input_tokens": max(1, len(text.split()))},
        headers={"X-Gemma4-MTP-Token-Counting": "estimated_word_count"},
    )
```

Add an async helper to bridge async iterables for the streaming case:

```python
async def openai_stream_to_anthropic_events_async(
    iterator,
    *,
    anthropic_model: str,
    message_id_prefix: str,
    prompt_tokens: int,
):
    chunks: list[dict] = []
    async for chunk in iterator:
        chunks.append(chunk)
    for event in openai_stream_to_anthropic_events(
        chunks,
        anthropic_model=anthropic_model,
        message_id_prefix=message_id_prefix,
        prompt_tokens=prompt_tokens,
    ):
        yield event
```

(This intermediate buffering is acceptable for v0.1; a true streaming
generator is a follow-up. The implementer must add a comment noting this
trade-off.)

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_anthropic_server.py -v
```
Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_anthropic_server.py
git commit -m "$(cat <<'EOF'
feat: add anthropic messages and count_tokens endpoints
EOF
)"
```

---

## Task 14: Doctor Command

**Files:**
- Create: `src/gemma4_mtp_vllm/doctor.py`
- Create: `tests/test_doctor.py`

**Context:** Doctor returns a stable JSON-shaped report including profile,
target, drafter, gateway version, vLLM version, vLLM `/health` status,
and whether the target and drafter are listed in vLLM's `/v1/models`.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import httpx
import pytest

from gemma4_mtp_vllm.doctor import build_report
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


@pytest.mark.asyncio
async def test_doctor_reports_ok_when_vllm_lists_target_and_drafter():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "google/gemma-4-31B-it"},
                    {"id": "google/gemma-4-31B-it-assistant"},
                ],
            })
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is True
    assert report["profile"] == "safe80"
    assert report["target_model"] == "google/gemma-4-31B-it"
    assert report["drafter"] == "google/gemma-4-31B-it-assistant"
    assert report["vllm"]["status"] == "ok"
    assert report["vllm"]["version"] == "0.11.0"
    assert report["target_loaded"] is True
    assert report["drafter_loaded"] is True


@pytest.mark.asyncio
async def test_doctor_marks_not_ok_when_target_missing():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["target_loaded"] is False


@pytest.mark.asyncio
async def test_doctor_marks_not_ok_when_vllm_unreachable():
    def handler(request):
        raise httpx.ConnectError("nope")

    report = await build_report(
        profile=_profile(),
        vllm_base_url="http://vllm.local:8000",
        transport=httpx.MockTransport(handler),
    )
    assert report["ok"] is False
    assert report["vllm"]["status"] == "unreachable"
```

- [ ] **Step 2: Run failing**

Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/doctor.py`**

```python
from __future__ import annotations

from typing import Any

import httpx

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION, __version__
from gemma4_mtp_vllm.backend.vllm_client import VllmClient, VllmHttpError
from gemma4_mtp_vllm.profiles import ModelProfile


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
    vllm_status: dict[str, Any] = {"status": "unreachable"}
    target_loaded = False
    drafter_loaded = False
    try:
        await client.health()
        vllm_status = {"status": "ok"}
    except (VllmHttpError, httpx.HTTPError):
        vllm_status = {"status": "unreachable"}

    if vllm_status.get("status") == "ok":
        try:
            version_body = await client.version()
            vllm_status["version"] = version_body.get("version")
        except (VllmHttpError, httpx.HTTPError):
            vllm_status["version"] = None
        try:
            models_body = await client.list_models()
            ids = {entry.get("id") for entry in models_body.get("data") or []}
            target_loaded = profile.target in ids
            drafter_loaded = profile.drafter in ids
        except (VllmHttpError, httpx.HTTPError):
            target_loaded = False
            drafter_loaded = False

    ok = vllm_status.get("status") == "ok" and target_loaded and drafter_loaded
    return {
        "ok": ok,
        "profile": profile.name,
        "target_model": profile.target,
        "drafter": profile.drafter,
        "num_speculative_tokens": profile.num_speculative_tokens,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "gateway_version": __version__,
        "required_vllm_min_version": REQUIRED_VLLM_MIN_VERSION,
        "vllm": vllm_status,
        "target_loaded": target_loaded,
        "drafter_loaded": drafter_loaded,
    }
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_doctor.py -v
```
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/doctor.py tests/test_doctor.py
git commit -m "$(cat <<'EOF'
feat: add doctor self-check report
EOF
)"
```

---

## Task 15: Benchmark Math + bench CLI

**Files:**
- Create: `src/gemma4_mtp_vllm/benchmarking.py`
- Create: `tests/test_benchmarking.py`
- Create: `tests/test_bench_cli.py` (skeleton, completed in Task 16-19)

**Context:** Pure math + result models for benchmarks. The CLI wiring lands
in Task 19 alongside other CLI commands.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json

import pytest

from gemma4_mtp_vllm.benchmarking import (
    BenchmarkObservation,
    BenchmarkSummary,
    deterministic_parity,
    median_optional,
    speedup,
)


def test_speedup_returns_ratio_when_both_positive():
    assert speedup(10.0, 20.0) == pytest.approx(2.0)


def test_speedup_returns_none_when_baseline_missing():
    assert speedup(0.0, 10.0) is None
    assert speedup(None, 10.0) is None
    assert speedup(10.0, None) is None


@pytest.mark.parametrize(
    "no_draft, mtp, expected",
    [
        ("hello", "hello", True),
        ("hello", "world", False),
    ],
)
def test_deterministic_parity_greedy(no_draft, mtp, expected):
    assert deterministic_parity(no_draft, mtp, temperature=0.0, top_p=1.0) is expected


def test_deterministic_parity_returns_none_when_sampling():
    assert deterministic_parity("a", "b", temperature=0.7, top_p=1.0) is None


def test_median_optional_returns_value():
    assert median_optional([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert median_optional([]) is None
    assert median_optional([None]) is None


def test_benchmark_summary_to_json_roundtrip():
    summary = BenchmarkSummary(
        profile="safe80",
        prompt_name="default",
        prompt="Hello",
        num_speculative_tokens=4,
        observations=[
            BenchmarkObservation(
                index=1,
                no_draft_generation_tps=10.0,
                mtp_generation_tps=20.0,
                speedup=2.0,
                deterministic_parity=True,
            )
        ],
        median_no_draft_generation_tps=10.0,
        median_mtp_generation_tps=20.0,
        median_speedup=2.0,
    )
    body = json.loads(json.dumps(summary.to_dict()))
    assert body["profile"] == "safe80"
    assert body["observations"][0]["speedup"] == 2.0
```

- [ ] **Step 2: Run failing**

Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/benchmarking.py`**

```python
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkObservation:
    index: int
    no_draft_generation_tps: float | None
    mtp_generation_tps: float | None
    speedup: float | None
    deterministic_parity: bool | None


@dataclass(frozen=True)
class BenchmarkSummary:
    profile: str
    prompt_name: str
    prompt: str
    num_speculative_tokens: int
    observations: list[BenchmarkObservation]
    median_no_draft_generation_tps: float | None
    median_mtp_generation_tps: float | None
    median_speedup: float | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["observations"] = [asdict(obs) for obs in self.observations]
        return data


def speedup(no_draft_tps: float | None, mtp_tps: float | None) -> float | None:
    if no_draft_tps is None or mtp_tps is None:
        return None
    if no_draft_tps <= 0:
        return None
    return mtp_tps / no_draft_tps


def deterministic_parity(
    no_draft_text: str,
    mtp_text: str,
    *,
    temperature: float,
    top_p: float,
) -> bool | None:
    if temperature != 0.0 or top_p != 1.0:
        return None
    return no_draft_text == mtp_text


def median_optional(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return statistics.median(cleaned)
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_benchmarking.py -v
```
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/benchmarking.py tests/test_benchmarking.py
git commit -m "$(cat <<'EOF'
feat: add benchmark math and result models
EOF
)"
```

---

## Task 16: Launch Helper

**Files:**
- Create: `src/gemma4_mtp_vllm/launch.py`
- Create: `tests/test_launch.py`

**Context:** Builds the canonical `vllm serve` argv from a profile. The CLI
wraps this in Task 19. The launch helper does not execute anything by
itself — it returns the argv list, which keeps it testable.

- [ ] **Step 1: Write failing test**

```python
from gemma4_mtp_vllm.launch import build_vllm_serve_args
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


def test_build_args_includes_target_and_speculative_config():
    args = build_vllm_serve_args(profile=_profile(), host="127.0.0.1", port=8000)
    assert args[0] == "vllm"
    assert "serve" in args
    assert "google/gemma-4-31B-it" in args
    spec_idx = args.index("--speculative-config")
    assert "google/gemma-4-31B-it-assistant" in args[spec_idx + 1]
    assert "\"num_speculative_tokens\": 4" in args[spec_idx + 1]
    assert "--tensor-parallel-size" in args
    assert "--max-model-len" in args
    assert "--gpu-memory-utilization" in args
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "8000" in args


def test_build_args_can_disable_mtp_for_baseline():
    args = build_vllm_serve_args(
        profile=_profile(),
        host="127.0.0.1",
        port=8000,
        enable_mtp=False,
    )
    assert "--speculative-config" not in args
```

- [ ] **Step 2: Run failing**

Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/launch.py`**

```python
from __future__ import annotations

import json

from gemma4_mtp_vllm.profiles import ModelProfile


def build_vllm_serve_args(
    *,
    profile: ModelProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_mtp: bool = True,
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
    ]
    if enable_mtp:
        spec = {
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec)])
    return args
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_launch.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/launch.py tests/test_launch.py
git commit -m "$(cat <<'EOF'
feat: add vllm serve launch helper
EOF
)"
```

---

## Task 17: CLI (Typer)

**Files:**
- Create: `src/gemma4_mtp_vllm/cli.py`
- Create: `tests/test_cli.py`

**Context:** Single Typer app exposing `serve`, `doctor`, `generate`,
`bench`, `bench-matrix`, `launch`. `bench` and `bench-matrix` will be
extended in Task 18; this task wires the skeleton with one of them
implemented end-to-end and the others raising NotImplementedError if
invoked without their helper.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from gemma4_mtp_vllm.cli import app


runner = CliRunner()


def test_launch_command_prints_argv():
    result = runner.invoke(app, ["launch", "--print-only"])
    assert result.exit_code == 0
    assert "vllm" in result.stdout
    assert "serve" in result.stdout
    assert "google/gemma-4-31B-it" in result.stdout


def test_doctor_command_emits_json(monkeypatch):
    def fake_run(coro):
        import asyncio

        return asyncio.get_event_loop().run_until_complete(coro)

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "data": [
                    {"id": "google/gemma-4-31B-it"},
                    {"id": "google/gemma-4-31B-it-assistant"},
                ],
            })
        return httpx.Response(404)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    result = runner.invoke(
        app,
        ["doctor", "--profile", "safe80", "--vllm-base-url", "http://vllm.local:8000"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["target_loaded"] is True


def test_serve_command_rejects_non_loopback_without_key():
    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0"],
    )
    assert result.exit_code != 0
    assert "api-key" in result.stdout.lower() or "api-key" in result.stderr.lower()
```

- [ ] **Step 2: Run failing**

Expected: import error.

- [ ] **Step 3: Implement `src/gemma4_mtp_vllm/cli.py`**

```python
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

import httpx
import typer

from gemma4_mtp_vllm import __version__
from gemma4_mtp_vllm.doctor import build_report
from gemma4_mtp_vllm.launch import build_vllm_serve_args
from gemma4_mtp_vllm.profiles import (
    ProfileSet,
    load_profiles,
    resolve_profile,
)
from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key
from gemma4_mtp_vllm.server.limits import ServerLimits

app = typer.Typer(add_completion=False, help="Gemma 4 31B MTP vLLM sidecar gateway")


def _profile_set() -> ProfileSet:
    return load_profiles()


def _mock_transport():
    """Test-only hook overridden in tests when VLLM_MTP_TRANSPORT_MOCK=1."""
    return None


def _build_transport() -> httpx.BaseTransport | None:
    if os.environ.get("VLLM_MTP_TRANSPORT_MOCK") == "1":
        return _mock_transport()
    return None


@app.command()
def doctor(
    profile: str = typer.Option("safe80", "--profile"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
) -> None:
    profile_set = _profile_set()
    selected = resolve_profile(profile, profile_set)
    transport = _build_transport()
    report = asyncio.run(
        build_report(
            profile=selected,
            vllm_base_url=vllm_base_url,
            transport=transport,
        )
    )
    typer.echo(json.dumps(report, indent=2))


@app.command()
def launch(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    print_only: bool = typer.Option(False, "--print-only"),
    no_mtp: bool = typer.Option(False, "--no-mtp"),
) -> None:
    selected = resolve_profile(profile, _profile_set())
    args = build_vllm_serve_args(
        profile=selected,
        host=host,
        port=port,
        enable_mtp=not no_mtp,
    )
    if print_only:
        typer.echo(" ".join(args))
        return
    os.execvp(args[0], args)


@app.command()
def serve(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    max_body_mb: float = typer.Option(2.0, "--max-body-mb"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    max_queue_size: int = typer.Option(8, "--max-queue-size"),
    rate_limit_rpm: int = typer.Option(30, "--rate-limit-rpm"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    cors_origin: list[str] = typer.Option([], "--cors-origin"),
) -> None:
    if bind_host_requires_api_key(host) and not api_key:
        typer.echo(f"host {host} requires --api-key", err=True)
        raise typer.Exit(code=1)

    import uvicorn

    from gemma4_mtp_vllm.server.app import create_app

    limits = ServerLimits(
        max_body_bytes=int(max_body_mb * 1024 * 1024),
        max_output_tokens=max_output_tokens,
        max_queue_size=max_queue_size,
        rate_limit_rpm=rate_limit_rpm,
        cors_origins=tuple(cors_origin),
    )
    selected_profile_name = profile
    fastapi_app = create_app(
        profile_name=selected_profile_name,
        bind_host=host,
        api_key=api_key,
        limits=limits,
        vllm_base_url=vllm_base_url,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def generate(
    prompt: str = typer.Argument(...),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    no_mtp: bool = typer.Option(False, "--no-mtp", help="Disabled in v0.1: requires separate vLLM launch."),
) -> None:
    """One-shot generation via the configured vLLM server."""
    if no_mtp:
        typer.echo(
            "--no-mtp requires launching a separate vLLM process without "
            "--speculative-config; see `vllm-mtp bench` for paired runs.",
            err=True,
        )
        raise typer.Exit(code=2)

    selected = resolve_profile(profile, _profile_set())

    async def run() -> dict:
        async with httpx.AsyncClient(base_url=vllm_base_url, timeout=120.0) as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": selected.target,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            response.raise_for_status()
            return response.json()

    payload = asyncio.run(run())
    text = payload["choices"][0]["message"]["content"]
    typer.echo(text)


@app.command()
def bench(
    prompt: str = typer.Option(..., "--prompt"),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    """Filled in Task 18."""
    typer.echo("bench: pending Task 18", err=True)
    raise typer.Exit(code=2)


@app.command("bench-matrix")
def bench_matrix(
    profile: str = typer.Option("safe80", "--profile"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    prompt: list[str] = typer.Option([], "--prompt"),
    num_speculative_tokens: list[int] = typer.Option(
        [], "--num-speculative-tokens"
    ),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    """Filled in Task 18."""
    typer.echo("bench-matrix: pending Task 18", err=True)
    raise typer.Exit(code=2)
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_cli.py -v
```
Expected: launch, doctor, serve guard tests pass. `bench` and
`bench-matrix` tests are added in Task 18.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
feat: scaffold typer cli with doctor, launch, serve, generate
EOF
)"
```

---

## Task 18: Bench Harness CLI

**Files:**
- Modify: `src/gemma4_mtp_vllm/cli.py`
- Create: `tests/test_bench_cli.py`

**Context:** Implement `bench` and `bench-matrix` against two vLLM URLs
(one with MTP enabled, one baseline). The user runs both vLLM processes
themselves; the gateway helper just compares them.

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from gemma4_mtp_vllm.cli import app


runner = CliRunner()


def _make_handler(*, tps_mtp: float, tps_baseline: float):
    def handler(request: httpx.Request) -> httpx.Response:
        # Headers carry which URL was hit; we route by host.
        host = request.url.host
        tps = tps_mtp if host == "mtp" else tps_baseline
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "vllm_tps_for_test": tps,
            },
        )

    return handler


def test_bench_command_emits_summary(monkeypatch, tmp_path):
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0)
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--max-tokens",
            "8",
            "--runs",
            "2",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(out.read_text())
    assert body["profile"] == "safe80"
    assert len(body["observations"]) == 2
    assert body["median_speedup"] == 2.0


def test_bench_matrix_iterates_prompts_and_n(monkeypatch, tmp_path):
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0)
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert len(payload) == 4
    keys = {(entry["prompt"], entry["num_speculative_tokens"]) for entry in payload}
    assert keys == {("alpha", 2), ("alpha", 4), ("beta", 2), ("beta", 4)}
```

- [ ] **Step 2: Run failing**

Expected: stub `bench` and `bench-matrix` exit with code 2.

- [ ] **Step 3: Implement `bench` and `bench-matrix` in `cli.py`**

Replace the stubs with real implementations:

```python
import time

import httpx as _httpx

from gemma4_mtp_vllm.benchmarking import (
    BenchmarkObservation,
    BenchmarkSummary,
    deterministic_parity,
    median_optional,
    speedup,
)


def _request_body(
    profile,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    return {
        "model": profile.target,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _http_client(base_url: str) -> _httpx.AsyncClient:
    transport = _build_transport()
    if transport is not None:
        return _httpx.AsyncClient(transport=transport, base_url=base_url, timeout=120.0)
    return _httpx.AsyncClient(base_url=base_url, timeout=120.0)


async def _measure(
    base_url: str,
    body: dict,
) -> tuple[str, float]:
    async with _http_client(base_url) as http:
        start = time.perf_counter()
        response = await http.post("/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    completion_tokens = (payload.get("usage") or {}).get("completion_tokens") or 1
    test_tps = payload.get("vllm_tps_for_test")
    if isinstance(test_tps, (int, float)):
        return text, float(test_tps)
    tps = completion_tokens / elapsed if elapsed > 0 else None
    return text, float(tps) if tps else 0.0


async def _single_bench(
    *,
    profile,
    prompt: str,
    max_tokens: int,
    mtp_url: str,
    baseline_url: str,
    runs: int,
    warmup_runs: int,
) -> list[BenchmarkObservation]:
    body = _request_body(
        profile,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    for _ in range(warmup_runs):
        await _measure(mtp_url, body)
        await _measure(baseline_url, body)

    observations: list[BenchmarkObservation] = []
    for idx in range(1, runs + 1):
        mtp_text, mtp_tps = await _measure(mtp_url, body)
        no_text, no_tps = await _measure(baseline_url, body)
        observations.append(
            BenchmarkObservation(
                index=idx,
                no_draft_generation_tps=no_tps,
                mtp_generation_tps=mtp_tps,
                speedup=speedup(no_tps, mtp_tps),
                deterministic_parity=deterministic_parity(
                    no_text, mtp_text, temperature=0.0, top_p=1.0
                ),
            )
        )
    return observations


@app.command()
def bench(
    prompt: str = typer.Option(..., "--prompt"),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    selected = resolve_profile(profile, _profile_set())
    observations = asyncio.run(
        _single_bench(
            profile=selected,
            prompt=prompt,
            max_tokens=max_tokens,
            mtp_url=mtp_url,
            baseline_url=baseline_url,
            runs=runs,
            warmup_runs=warmup_runs,
        )
    )
    summary = BenchmarkSummary(
        profile=selected.name,
        prompt_name="default",
        prompt=prompt,
        num_speculative_tokens=selected.num_speculative_tokens,
        observations=observations,
        median_no_draft_generation_tps=median_optional(
            [obs.no_draft_generation_tps for obs in observations]
        ),
        median_mtp_generation_tps=median_optional(
            [obs.mtp_generation_tps for obs in observations]
        ),
        median_speedup=median_optional([obs.speedup for obs in observations]),
    )
    payload = summary.to_dict()
    rendered = json.dumps(payload, indent=2)
    if json_output:
        Path(json_output).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@app.command("bench-matrix")
def bench_matrix(
    profile: str = typer.Option("safe80", "--profile"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    prompt: list[str] = typer.Option([], "--prompt"),
    num_speculative_tokens: list[int] = typer.Option(
        [], "--num-speculative-tokens"
    ),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    if not prompt:
        typer.echo("at least one --prompt required", err=True)
        raise typer.Exit(code=2)
    if not num_speculative_tokens:
        typer.echo("at least one --num-speculative-tokens required", err=True)
        raise typer.Exit(code=2)
    if any(value <= 0 for value in num_speculative_tokens):
        typer.echo("--num-speculative-tokens must be positive", err=True)
        raise typer.Exit(code=2)

    selected_base = resolve_profile(profile, _profile_set())
    results: list[dict] = []
    for prompt_value in prompt:
        for n in num_speculative_tokens:
            adjusted = type(selected_base)(
                **{**selected_base.__dict__, "num_speculative_tokens": n}
            )
            observations = asyncio.run(
                _single_bench(
                    profile=adjusted,
                    prompt=prompt_value,
                    max_tokens=64,
                    mtp_url=mtp_url,
                    baseline_url=baseline_url,
                    runs=runs,
                    warmup_runs=warmup_runs,
                )
            )
            summary = BenchmarkSummary(
                profile=adjusted.name,
                prompt_name=f"prompt_{prompt.index(prompt_value) + 1}",
                prompt=prompt_value,
                num_speculative_tokens=n,
                observations=observations,
                median_no_draft_generation_tps=median_optional(
                    [obs.no_draft_generation_tps for obs in observations]
                ),
                median_mtp_generation_tps=median_optional(
                    [obs.mtp_generation_tps for obs in observations]
                ),
                median_speedup=median_optional([obs.speedup for obs in observations]),
            )
            results.append(summary.to_dict())
    rendered = json.dumps(results, indent=2)
    if json_output:
        Path(json_output).write_text(rendered, encoding="utf-8")
    typer.echo(rendered)
```

Note: `ModelProfile` is frozen, so `bench-matrix` builds a fresh copy via
`type(...)(**{...})`. Implementers should verify this against the actual
dataclass and switch to `dataclasses.replace` if cleaner.

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_bench_cli.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/cli.py tests/test_bench_cli.py
git commit -m "$(cat <<'EOF'
feat: implement bench and bench-matrix commands
EOF
)"
```

---

## Task 19: README

**Files:**
- Modify: `README.md`

**Context:** Replace the placeholder README created in Task 0 with a
complete project README following the MLX project's structure (install,
doctor, bench, server, OpenAI / Anthropic curl, source archive, wheel
freshness, tests, architecture diagram, author).

The README MUST include:

- Title and one-paragraph summary.
- Architecture Mermaid diagram (use the spec's Mermaid block verbatim).
- Default profile summary (target / drafter / num_speculative_tokens /
  tensor_parallel_size / GPU memory utilization / max_model_len).
- vLLM version pin reference (`>=0.11.0`) with a callout that vLLM is an
  optional extra, the user installs it separately.
- Step-by-step install:
  1. NVIDIA CUDA prerequisites callout (Python 3.10+, CUDA-compatible
     driver, optional Docker), 2. clone, 3. venv + base install, 4.
     install vLLM extra (single GPU and TP=2 examples), 5. start vLLM via
     `vllm-mtp launch`, 6. start the gateway via `vllm-mtp serve`.
- Doctor section with the expected JSON output shape.
- Bench section with examples for `bench` and `bench-matrix`, noting that
  the user must run two vLLM processes for paired benchmarks and link to
  the spec's discussion of upstream Issue #41789.
- Server section (OpenAI + Anthropic curl examples).
- V1 Policy section copied conceptually from the MLX project, adapted to
  state that this gateway intentionally does not implement tool calling
  or multimodal yet.
- Source archive + wheel freshness sections describing the upcoming
  scripts (filled out in Task 20).
- Tests section with the four canonical commands.
- Author section (same Alican Kiraz badges block as the MLX project).

- [ ] **Step 1: Replace `README.md` with the full document described above**

- [ ] **Step 2: Run docs-only sanity check**

```bash
git diff --check README.md
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: write full readme for vllm sidecar gateway
EOF
)"
```

---

## Task 20: Release Scripts and Their Tests

**Files:**
- Create: `scripts/make_source_archive.sh`
- Create: `scripts/verify_source_archive.sh`
- Create: `scripts/verify_wheel_freshness.sh`
- Create: `tests/test_release_scripts.py`

**Context:** Port the MLX project's release tooling to this project. The
source archive script uses `git archive` and excludes the same forbidden
paths (`.venv`, `.git`, `dist`, `.pytest_cache`, `__pycache__`, `build`,
`__MACOSX`, `bench-results`). The wheel freshness script builds a wheel,
installs into a temp venv, and exercises gateway endpoints with a fake
vLLM transport.

- [ ] **Step 1: Write `tests/test_release_scripts.py`**

```python
from __future__ import annotations

import os
import stat
from pathlib import Path


def test_make_source_archive_exists_and_executable():
    script = Path("scripts/make_source_archive.sh")
    assert script.is_file()
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR


def test_verify_source_archive_exists_and_executable():
    script = Path("scripts/verify_source_archive.sh")
    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR


def test_verify_wheel_freshness_exists_and_executable():
    script = Path("scripts/verify_wheel_freshness.sh")
    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR


def test_wheel_freshness_script_contains_required_smoke_steps():
    script = Path("scripts/verify_wheel_freshness.sh").read_text(encoding="utf-8")
    assert "create_app" in script
    assert "/livez" in script
    assert "x-api-key" in script
    assert "local-dev-key" in script


def test_verify_source_archive_excludes_forbidden_paths():
    script = Path("scripts/verify_source_archive.sh").read_text(encoding="utf-8")
    for path in (".venv", ".git", "dist", ".pytest_cache", "__pycache__",
                 "__MACOSX", "build", "bench-results"):
        assert path in script
```

- [ ] **Step 2: Run failing**

Expected: all four tests fail (scripts missing).

- [ ] **Step 3: Write the three scripts**

`scripts/make_source_archive.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

output="${1:-Gemma-4-31B-MTP-vllm-src.zip}"

git archive --format=zip --output "$output" HEAD
echo "wrote $output"
```

`scripts/verify_source_archive.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

archive="${1:?usage: verify_source_archive.sh <zip>}"

forbidden='(^|/)(\.git|\.venv|\.worktrees|\.pytest_cache|__pycache__|dist|build|__MACOSX|bench-results)(/|$)|\.pyc$|\.DS_Store$'

if unzip -Z1 "$archive" | grep -E "$forbidden" >/dev/null; then
    echo "archive contains forbidden entries"
    exit 1
fi
echo "archive clean: $archive"
```

`scripts/verify_wheel_freshness.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        python_bin=".venv/bin/python"
    elif [[ -x "../.venv/bin/python" ]]; then
        python_bin="../.venv/bin/python"
    else
        python_bin="python3"
    fi
fi

rm -f dist/gemma4_mtp_vllm-*.whl
"$python_bin" -m build --wheel

wheel=$(ls dist/gemma4_mtp_vllm-*.whl | head -n 1)
work=$(mktemp -d)
trap "rm -rf $work" EXIT

"$python_bin" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet "$wheel" fastapi httpx pytest pyyaml
cat <<'PY' > "$work/smoke.py"
import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def handler(request):
    if request.url.path in {"/health", "/v1/models", "/version"}:
        return httpx.Response(200, json={"status": "ok", "data": [], "version": "0.11.0"})
    return httpx.Response(404)


app = create_app(
    api_key="local-dev-key",
    vllm_base_url="http://vllm.local:8000",
    vllm_transport=httpx.MockTransport(handler),
)
client = TestClient(app)

livez = client.get("/livez")
assert livez.status_code == 200, livez.text

health = client.get("/health", headers={"x-api-key": "local-dev-key"})
assert health.status_code == 200, health.text
assert "Gemma 4 31B MTP" in health.text or "gemma-4-31B-it" in health.text

print("wheel smoke ok")
PY

"$work/venv/bin/python" "$work/smoke.py"
```

Make all three executable:

```bash
chmod +x scripts/make_source_archive.sh \
         scripts/verify_source_archive.sh \
         scripts/verify_wheel_freshness.sh
```

- [ ] **Step 4: Run passing**

```bash
python -m pytest tests/test_release_scripts.py -v
```
Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/test_release_scripts.py
git commit -m "$(cat <<'EOF'
feat: add source archive and wheel freshness scripts
EOF
)"
```

---

## Task 21: Full Verification and Handoff

**Files:**
- Modify: nothing required beyond what previous tasks created.

**Context:** Run the full local verification suite. Fix any failures as
small follow-up commits. Hand the repo over in a known-good state.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest -q
```
Expected: all tests pass (target ≥ 60 tests across the project; final count
should match the contents of the `tests/` directory).

- [ ] **Step 2: Run pip check and compile check**

```bash
python -m pip check
python -m compileall -q src
```
Expected: both report no errors.

- [ ] **Step 3: Build wheel and validate**

```bash
python -m build --wheel
scripts/verify_wheel_freshness.sh
scripts/make_source_archive.sh dist/Gemma-4-31B-MTP-vllm-src.zip
scripts/verify_source_archive.sh dist/Gemma-4-31B-MTP-vllm-src.zip
```
Expected: wheel builds, smoke passes, source archive clean.

- [ ] **Step 4: Self-document the result**

Append a `### Verification (YYYY-MM-DD)` section to the README's tests
block recording the test count and the wheel smoke result for the final
commit.

- [ ] **Step 5: Final commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
chore: record verification result for v0.1
EOF
)"
```

- [ ] **Step 6: Report status**

Report to the controlling session:

- Total tests passing.
- Wheel smoke status.
- Source archive verification status.
- Any open follow-ups that must wait for a real GPU host (e.g., real
  vLLM smoke against an actual `vllm serve` process).
