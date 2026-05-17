# v0.1 Release Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gemma 4 31B MTP vLLM sidecar safe to publish as a public alpha by fixing the verified release blockers around vLLM versioning, MTP launch config, real vLLM health semantics, request validation, admission control, CORS, doctor semantics, and release documentation.

**Architecture:** Keep vLLM as an external OpenAI-compatible server and keep this package as the sidecar gateway. Add small focused modules for version checks and request validation, wire existing runtime admission into the generation paths, and make diagnostics report what is actually observable from vLLM 0.21.0. Do not implement tool calling, multimodal handling, or full Anthropic true streaming in this release blocker pass.

**Tech Stack:** Python 3.10+, FastAPI, HTTPX, Typer, Pydantic-free strict validators, Pytest, vLLM 0.21.0, POSIX shell release scripts.

---

## Verified Assessment

The external review findings were checked against the repository and the 2x RTX 5090 remote smoke run. These items are confirmed:

- `pyproject.toml` and `src/gemma4_mtp_vllm/__init__.py` still say `vllm>=0.11.0`, but Gemma 4 MTP works in vLLM `0.21.0` and `0.20.2` rejects `method="mtp"`.
- `src/gemma4_mtp_vllm/launch.py` emits `--speculative-config` without `"method":"mtp"`.
- The real vLLM `GET /health` endpoint returns `200` with an empty body, so `VllmClient.health()` currently raises `JSONDecodeError` and gateway `/health` returns `500`.
- `doctor.py` requires the drafter to appear in `/v1/models`, but real vLLM lists the served target model only.
- `RuntimeState.acquire_generation_slot()` exists and is tested, but generation endpoints never call it.
- CORS preflight reaches auth and returns `401` instead of a CORS response.
- Typed-bad JSON can produce `500` or pass through to vLLM, for example `model: []`, `max_tokens: "abc"`, and `messages: "abc"`.
- Anthropic boundary errors from middleware are OpenAI-shaped.
- `bench-matrix` changes local metadata for `num_speculative_tokens`, but vLLM treats that value as server startup config.

Remote evidence from the 2x RTX 5090 host:

```text
vLLM 0.20.2 + method=mtp: Unsupported speculative method: 'mtp'
vLLM 0.21.0 + method=mtp: server started on 127.0.0.1:8010
direct vLLM chat: 200
gateway OpenAI chat: 200
gateway completions: 200
gateway Anthropic messages: 200
gateway count_tokens: 200
gateway /health: 500 because vLLM /health body is empty
MTP metrics after smoke: drafts=17, draft_tokens=68, accepted_tokens=35
```

## File Structure

Modify these existing files:

- `pyproject.toml` - vLLM optional dependency range.
- `src/gemma4_mtp_vllm/__init__.py` - required vLLM minimum version constant.
- `src/gemma4_mtp_vllm/launch.py` - explicit MTP method and optional reasoning parser launch args.
- `src/gemma4_mtp_vllm/cli.py` - shell-safe `launch --print-only`, safer `bench-matrix`, and updated output semantics.
- `src/gemma4_mtp_vllm/backend/vllm_client.py` - tolerate empty `200` JSON body for `/health`, expose metrics text.
- `src/gemma4_mtp_vllm/doctor.py` - version-aware report, no drafter false negative.
- `src/gemma4_mtp_vllm/server/app.py` - validation, slot admission, queue errors, health capability truthfulness.
- `src/gemma4_mtp_vllm/server/middleware.py` - CORS preflight before auth, protocol-aware boundary errors.
- `src/gemma4_mtp_vllm/server/runtime_state.py` - streaming request counter if used in metrics.
- `src/gemma4_mtp_vllm/backend/response_parser.py` - non-stream visible text fallback stripping.
- `README.md` - vLLM 0.21.0 requirement, source archive hygiene, real hardware verification notes.
- `scripts/verify_wheel_freshness.sh` - fake vLLM version fixture.

Create these new files:

- `src/gemma4_mtp_vllm/versioning.py` - semantic version parsing and comparison for `major.minor.patch`.
- `src/gemma4_mtp_vllm/server/validation.py` - strict request validators for OpenAI chat, OpenAI completions, Anthropic messages, and Anthropic count tokens.
- `tests/test_validation.py` - validator unit tests.
- `tests/test_versioning.py` - version comparison tests.

Modify these test files:

- `tests/test_package_metadata.py`
- `tests/test_launch.py`
- `tests/test_cli.py`
- `tests/test_doctor.py`
- `tests/test_vllm_client.py`
- `tests/test_server_app.py`
- `tests/test_server_health.py`
- `tests/test_middleware.py`
- `tests/test_openai_server.py`
- `tests/test_anthropic_server.py`
- `tests/test_runtime_state.py`
- `tests/test_bench_cli.py`
- `tests/test_release_scripts.py`

---

## P0 Blocker Tasks

### Task 1: vLLM 0.21.0 Requirement and Explicit MTP Launch

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/gemma4_mtp_vllm/__init__.py`
- Modify: `src/gemma4_mtp_vllm/launch.py`
- Modify: `src/gemma4_mtp_vllm/cli.py`
- Modify: `tests/test_package_metadata.py`
- Modify: `tests/test_launch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for vLLM version and launch config**

Add these assertions to `tests/test_package_metadata.py`:

```python
def test_required_vllm_minimum_matches_gemma4_mtp_release():
    import gemma4_mtp_vllm

    assert gemma4_mtp_vllm.REQUIRED_VLLM_MIN_VERSION == "0.21.0"
```

Add these assertions to `tests/test_launch.py`:

```python
import json
import shlex


def test_build_args_includes_explicit_mtp_method():
    args = build_vllm_serve_args(profile=_profile(), host="127.0.0.1", port=8000)
    spec_idx = args.index("--speculative-config")
    spec = json.loads(args[spec_idx + 1])

    assert spec == {
        "method": "mtp",
        "model": "google/gemma-4-31B-it-assistant",
        "num_speculative_tokens": 4,
    }


def test_build_args_include_gemma4_reasoning_parser():
    args = build_vllm_serve_args(profile=_profile(), host="127.0.0.1", port=8000)

    assert "--reasoning-parser" in args
    assert args[args.index("--reasoning-parser") + 1] == "gemma4"


def test_print_only_command_is_shell_safe():
    args = build_vllm_serve_args(profile=_profile(), host="127.0.0.1", port=8000)
    rendered = shlex.join(args)

    assert "'{\" in rendered
    assert shlex.split(rendered) == args
```

Add a CLI test in `tests/test_cli.py`:

```python
def test_launch_print_only_uses_shell_safe_rendering():
    result = runner.invoke(app, ["launch", "--profile", "safe80", "--print-only"])

    assert result.exit_code == 0
    assert "--speculative-config" in result.output
    assert "'{\" in result.output
    assert "\"method\":\"mtp\"" in result.output
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
python -m pytest tests/test_package_metadata.py tests/test_launch.py tests/test_cli.py -q
```

Expected failure:

```text
FAILED ... REQUIRED_VLLM_MIN_VERSION == "0.21.0"
FAILED ... spec["method"]
FAILED ... --reasoning-parser
FAILED ... shell-safe rendering
```

- [ ] **Step 3: Update version constants and optional dependency**

Change `src/gemma4_mtp_vllm/__init__.py` to:

```python
__version__ = "0.1.0"
REQUIRED_VLLM_MIN_VERSION = "0.21.0"

__all__ = ["REQUIRED_VLLM_MIN_VERSION", "__version__"]
```

Change the `vllm` extra in `pyproject.toml` to:

```toml
vllm = [
    "vllm>=0.21.0,<0.22.0",
]
```

- [ ] **Step 4: Update launch argument builder**

Change `src/gemma4_mtp_vllm/launch.py` to:

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
        "--reasoning-parser",
        "gemma4",
    ]
    if enable_mtp:
        spec = {
            "method": "mtp",
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec, separators=(",", ":"))])
    return args
```

- [ ] **Step 5: Render `launch --print-only` with `shlex.join`**

Add `import shlex` to `src/gemma4_mtp_vllm/cli.py` and change the print-only block:

```python
    if print_only:
        typer.echo(shlex.join(args))
        return
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/test_package_metadata.py tests/test_launch.py tests/test_cli.py -q
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/gemma4_mtp_vllm/__init__.py src/gemma4_mtp_vllm/launch.py src/gemma4_mtp_vllm/cli.py tests/test_package_metadata.py tests/test_launch.py tests/test_cli.py
git commit -m "fix: require vllm 0.21 for gemma4 mtp launch"
```

### Task 2: vLLM Client Health Semantics and Version Comparison

**Files:**
- Create: `src/gemma4_mtp_vllm/versioning.py`
- Modify: `src/gemma4_mtp_vllm/backend/vllm_client.py`
- Modify: `tests/test_versioning.py`
- Modify: `tests/test_vllm_client.py`

- [ ] **Step 1: Add version comparison tests**

Create `tests/test_versioning.py`:

```python
from gemma4_mtp_vllm.versioning import version_at_least


def test_version_at_least_accepts_required_version():
    assert version_at_least("0.21.0", "0.21.0")


def test_version_at_least_accepts_newer_patch():
    assert version_at_least("0.21.1", "0.21.0")


def test_version_at_least_rejects_older_minor():
    assert not version_at_least("0.20.2", "0.21.0")


def test_version_at_least_handles_local_suffix():
    assert version_at_least("0.21.0+cu129", "0.21.0")


def test_version_at_least_rejects_missing_version():
    assert not version_at_least(None, "0.21.0")
```

Add this test to `tests/test_vllm_client.py`:

```python
async def test_health_accepts_empty_success_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.local",
    ) as http:
        client = VllmClient(http=http, base_url="http://vllm.local")
        assert await client.health() == {"status": "ok"}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_versioning.py tests/test_vllm_client.py -q
```

Expected failure:

```text
ModuleNotFoundError: No module named 'gemma4_mtp_vllm.versioning'
JSONDecodeError for empty health body
```

- [ ] **Step 3: Implement version comparison helper**

Create `src/gemma4_mtp_vllm/versioning.py`:

```python
from __future__ import annotations

import re


_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def _parse_version(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = _VERSION_RE.match(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def version_at_least(value: str | None, minimum: str) -> bool:
    parsed_value = _parse_version(value)
    parsed_minimum = _parse_version(minimum)
    if parsed_value is None or parsed_minimum is None:
        return False
    return parsed_value >= parsed_minimum
```

- [ ] **Step 4: Make `VllmClient.health()` tolerate empty body**

Change `src/gemma4_mtp_vllm/backend/vllm_client.py`:

```python
    async def health(self) -> dict:
        response = await self._http.get("/health")
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        if not response.content:
            return {"status": "ok"}
        return response.json()
```

Keep `_json_or_raise()` strict for non-health JSON endpoints.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_versioning.py tests/test_vllm_client.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/versioning.py src/gemma4_mtp_vllm/backend/vllm_client.py tests/test_versioning.py tests/test_vllm_client.py
git commit -m "fix: handle real vllm health responses"
```

### Task 3: Doctor Semantics Without Drafter False Negative

**Plan correction (2026-05-17):** `target_served` must require
`profile.target` in `/v1/models`. Accepting `profile.name` would create a false
positive because `launch.py` serves `profile.target`, and gateway upstream
requests send `profile.target`.

**Files:**
- Modify: `src/gemma4_mtp_vllm/doctor.py`
- Modify: `tests/test_doctor.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing tests for real vLLM model listing**

Add to `tests/test_doctor.py`:

```python
@pytest.mark.asyncio
async def test_doctor_ok_when_target_served_and_drafter_not_listed():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"")
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "google/gemma-4-31B-it", "object": "model"}],
                },
            )
        return httpx.Response(404)

    profile = resolve_profile("safe80", load_profiles())
    report = await build_report(
        profile=profile,
        vllm_base_url="http://vllm.local",
        transport=httpx.MockTransport(handler),
    )

    assert report["ok"] is True
    assert report["version_ok"] is True
    assert report["target_served"] is True
    assert report["drafter_configured"] == "google/gemma-4-31B-it-assistant"
    assert report["drafter_loaded"] == "unknown"
```

Add:

```python
@pytest.mark.asyncio
async def test_doctor_rejects_old_vllm_version():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"")
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.20.2"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "google/gemma-4-31B-it"}]},
            )
        return httpx.Response(404)

    profile = resolve_profile("safe80", load_profiles())
    report = await build_report(
        profile=profile,
        vllm_base_url="http://vllm.local",
        transport=httpx.MockTransport(handler),
    )

    assert report["ok"] is False
    assert report["version_ok"] is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_doctor.py -q
```

Expected failure:

```text
report["ok"] is False when drafter is not listed
KeyError: 'version_ok'
```

- [ ] **Step 3: Update doctor report**

Change `_build_report()` in `src/gemma4_mtp_vllm/doctor.py` to use this shape:

```python
from gemma4_mtp_vllm.versioning import version_at_least


async def _build_report(
    *,
    client: VllmClient,
    profile: ModelProfile,
) -> dict[str, Any]:
    vllm_status: dict[str, Any] = {"status": "unreachable", "version": None}
    target_served = False

    try:
        await client.health()
        vllm_status = {"status": "ok", "version": None}
    except (VllmHttpError, httpx.HTTPError):
        vllm_status = {"status": "unreachable", "version": None}

    if vllm_status.get("status") == "ok":
        try:
            version_body = await client.version()
            vllm_status["version"] = version_body.get("version")
        except (VllmHttpError, httpx.HTTPError):
            vllm_status["version"] = None
        try:
            models_body = await client.list_models()
            ids = {entry.get("id") for entry in models_body.get("data") or []}
            target_served = profile.target in ids
        except (VllmHttpError, httpx.HTTPError):
            target_served = False

    version_ok = version_at_least(
        vllm_status.get("version"),
        REQUIRED_VLLM_MIN_VERSION,
    )
    ok = vllm_status.get("status") == "ok" and version_ok and target_served
    return {
        "ok": ok,
        "profile": profile.name,
        "target_model": profile.target,
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
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_doctor.py tests/test_cli.py -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_mtp_vllm/doctor.py tests/test_doctor.py tests/test_cli.py
git commit -m "fix: align doctor with real vllm mtp serving"
```

### Task 4: Strict Request Validation

**Files:**
- Create: `src/gemma4_mtp_vllm/server/validation.py`
- Create: `tests/test_validation.py`
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_openai_server.py`
- Modify: `tests/test_anthropic_server.py`

- [ ] **Step 1: Add validator unit tests**

Create `tests/test_validation.py`:

```python
import pytest

from gemma4_mtp_vllm.server.validation import (
    RequestValidationError,
    validate_anthropic_messages_payload,
    validate_openai_chat_payload,
    validate_openai_completions_payload,
)


def test_openai_chat_rejects_non_string_model():
    with pytest.raises(RequestValidationError) as exc:
        validate_openai_chat_payload({"model": [], "messages": []})

    assert exc.value.code == "invalid_request"
    assert "model must be a string" in exc.value.message


def test_openai_chat_rejects_non_list_messages():
    with pytest.raises(RequestValidationError) as exc:
        validate_openai_chat_payload({"model": "gemma", "messages": "abc"})

    assert "messages must be a non-empty list" in exc.value.message


def test_openai_chat_rejects_string_temperature():
    with pytest.raises(RequestValidationError) as exc:
        validate_openai_chat_payload(
            {
                "model": "gemma",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "0",
            }
        )

    assert "temperature must be a number" in exc.value.message


def test_openai_completions_rejects_bad_max_tokens():
    with pytest.raises(RequestValidationError) as exc:
        validate_openai_completions_payload(
            {"model": "gemma", "prompt": "hi", "max_tokens": "abc"}
        )

    assert "max_tokens must be a positive integer" in exc.value.message


def test_anthropic_messages_rejects_missing_max_tokens():
    with pytest.raises(RequestValidationError) as exc:
        validate_anthropic_messages_payload(
            {"model": "gemma", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert "max_tokens must be a positive integer" in exc.value.message
```

Add integration tests:

```python
def test_chat_completion_rejects_bad_message_shape():
    response = _client().post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={"model": "gemma-4-31b-mtp", "messages": "abc"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
```

```python
def test_anthropic_messages_rejects_bad_max_tokens():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "claude-gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": "bad",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_validation.py tests/test_openai_server.py tests/test_anthropic_server.py -q
```

Expected failure:

```text
ModuleNotFoundError for validation module
bad typed payloads return 500 or 200
```

- [ ] **Step 3: Implement validation module**

Create `src/gemma4_mtp_vllm/server/validation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestValidationError(Exception):
    message: str
    code: str = "invalid_request"
    status_code: int = 400


def _require_model(payload: dict[str, Any]) -> None:
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise RequestValidationError("model must be a string")


def _require_positive_int(payload: dict[str, Any], field: str) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RequestValidationError(f"{field} must be a positive integer")


def _optional_positive_int(payload: dict[str, Any], field: str) -> None:
    if field not in payload or payload[field] is None:
        return
    _require_positive_int(payload, field)


def _optional_number(payload: dict[str, Any], field: str, *, minimum: float | None = None, maximum: float | None = None) -> None:
    if field not in payload or payload[field] is None:
        return
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(f"{field} must be a number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise RequestValidationError(f"{field} must be at least {minimum:g}")
    if maximum is not None and numeric > maximum:
        raise RequestValidationError(f"{field} must be at most {maximum:g}")


def _optional_bool(payload: dict[str, Any], field: str) -> None:
    if field in payload and not isinstance(payload[field], bool):
        raise RequestValidationError(f"{field} must be a boolean")


def _content_is_text(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in value
        )
    return False


def _validate_messages(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise RequestValidationError("messages must be a non-empty list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise RequestValidationError(f"messages[{index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise RequestValidationError(f"messages[{index}].role must be a string")
        if not _content_is_text(message.get("content")):
            raise RequestValidationError(
                f"messages[{index}].content must be text or text blocks"
            )


def validate_openai_chat_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _validate_messages(payload.get("messages"))
    _optional_positive_int(payload, "max_tokens")
    _optional_number(payload, "temperature", minimum=0.0)
    _optional_number(payload, "top_p", minimum=0.0, maximum=1.0)
    _optional_positive_int(payload, "top_k")
    _optional_bool(payload, "stream")


def validate_openai_completions_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    prompt = payload.get("prompt")
    if not isinstance(prompt, (str, list)) or prompt == "":
        raise RequestValidationError("prompt must be a string or list")
    if isinstance(prompt, list) and not all(isinstance(item, str) for item in prompt):
        raise RequestValidationError("prompt list items must be strings")
    _optional_positive_int(payload, "max_tokens")
    _optional_number(payload, "temperature", minimum=0.0)
    _optional_number(payload, "top_p", minimum=0.0, maximum=1.0)


def validate_anthropic_messages_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _require_positive_int(payload, "max_tokens")
    _validate_messages(payload.get("messages"))
    system = payload.get("system")
    if system is not None and not _content_is_text(system):
        raise RequestValidationError("system must be text or text blocks")
    _optional_number(payload, "temperature", minimum=0.0)
    _optional_number(payload, "top_p", minimum=0.0, maximum=1.0)
    _optional_positive_int(payload, "top_k")
    _optional_bool(payload, "stream")


def validate_anthropic_count_tokens_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _validate_messages(payload.get("messages"))
    system = payload.get("system")
    if system is not None and not _content_is_text(system):
        raise RequestValidationError("system must be text or text blocks")
```

- [ ] **Step 4: Wire validation into app endpoints**

Import validators in `src/gemma4_mtp_vllm/server/app.py`:

```python
from gemma4_mtp_vllm.server.validation import (
    RequestValidationError,
    validate_anthropic_count_tokens_payload,
    validate_anthropic_messages_payload,
    validate_openai_chat_payload,
    validate_openai_completions_payload,
)
```

Add helper:

```python
def _validation_error_response(
    exc: RequestValidationError,
    *,
    protocol: str = "openai",
) -> JSONResponse:
    return protocol_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        protocol=protocol,
    )
```

After `_bounded_json()` in each endpoint:

```python
        try:
            validate_openai_chat_payload(payload)
            validate_openai_request(payload, mtp_enabled=True)
        except RequestValidationError as exc:
            return _validation_error_response(exc)
        except UnsupportedFeature as exc:
            return protocol_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
            )
```

Use `validate_openai_completions_payload(payload)` in `/v1/completions`, `validate_anthropic_messages_payload(payload)` in `/v1/messages`, and `validate_anthropic_count_tokens_payload(payload)` in `/v1/messages/count_tokens`.

Change `_alias_known()`:

```python
def _alias_known(value: Any, aliases: Iterable[str]) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value in set(aliases)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_validation.py tests/test_openai_server.py tests/test_anthropic_server.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/server/validation.py src/gemma4_mtp_vllm/server/app.py tests/test_validation.py tests/test_openai_server.py tests/test_anthropic_server.py
git commit -m "fix: validate request payloads before forwarding"
```

### Task 5: Bounded Admission and Generation Metrics

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/runtime_state.py`
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `tests/test_openai_server.py`
- Modify: `tests/test_anthropic_server.py`
- Modify: `tests/test_server_metrics.py`

- [ ] **Step 1: Add failing tests for total request and queue full behavior**

Add to `tests/test_openai_server.py`:

```python
def test_chat_completion_increments_total_requests():
    client = _client()
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )
    metrics = client.get("/metrics", headers={"x-api-key": "secret"}).text

    assert response.status_code == 200
    assert "gemma4_mtp_total_requests 1" in metrics
```

Add a queue rejection test in `tests/test_server_app.py`:

```python
def test_generation_queue_full_returns_429(monkeypatch):
    app = create_app(
        api_key="secret",
        limits=ServerLimits(max_queue_size=1),
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(_handler),
    )
    app.state.runtime_state._active = 2
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "queue_full"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_openai_server.py tests/test_server_app.py tests/test_server_metrics.py -q
```

Expected failure:

```text
gemma4_mtp_total_requests remains 0
queue full request reaches backend or returns a different error
```

- [ ] **Step 3: Add app helpers for slots**

Import `QueueFull` in `src/gemma4_mtp_vllm/server/app.py`:

```python
from gemma4_mtp_vllm.server.runtime_state import QueueFull, RuntimeState
```

Add:

```python
async def _acquire_slot_or_error(
    runtime_state: RuntimeState,
    *,
    protocol: str = "openai",
) -> Any | JSONResponse:
    try:
        return await runtime_state.acquire_generation_slot()
    except QueueFull:
        return protocol_error_response(
            status_code=429,
            code="queue_full",
            message="too many queued generation requests",
            protocol=protocol,
        )
```

- [ ] **Step 4: Wire non-streaming endpoints**

For `/v1/chat/completions`, after validation and alias check:

```python
        slot = await _acquire_slot_or_error(runtime_state)
        if isinstance(slot, JSONResponse):
            return slot
        try:
            response = await vllm.chat_completion(body)
        except VllmHttpError as exc:
            runtime_state.record_backend_error("vllm_http_error")
            return protocol_error_response(
                status_code=503,
                code="backend_unavailable",
                message=f"vllm returned {exc.status_code}",
            )
        finally:
            slot.release()
```

Apply the same structure to `/v1/completions` and `/v1/messages`, using `protocol="anthropic"` for `/v1/messages`.

- [ ] **Step 5: Wire streaming endpoints with generator finally**

For OpenAI streaming:

```python
            slot = await _acquire_slot_or_error(runtime_state)
            if isinstance(slot, JSONResponse):
                return slot

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
                finally:
                    slot.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream")
```

For Anthropic streaming, use the same `finally: slot.release()` pattern and `protocol="anthropic"` for the queue error.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/test_runtime_state.py tests/test_openai_server.py tests/test_anthropic_server.py tests/test_server_metrics.py tests/test_server_app.py -q
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/gemma4_mtp_vllm/server/runtime_state.py src/gemma4_mtp_vllm/server/app.py tests/test_runtime_state.py tests/test_openai_server.py tests/test_anthropic_server.py tests/test_server_metrics.py tests/test_server_app.py
git commit -m "fix: enforce bounded generation admission"
```

### Task 6: CORS Preflight and Protocol-Aware Boundary Errors

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/middleware.py`
- Modify: `tests/test_middleware.py`
- Modify: `tests/test_anthropic_server.py`

- [ ] **Step 1: Add failing tests for preflight and Anthropic errors**

Add to `tests/test_middleware.py`:

```python
def test_cors_preflight_allowed_origin_bypasses_auth_and_rate_limit():
    limits = ServerLimits(cors_origins=("https://app.test",), rate_limit_rpm=1)
    client = TestClient(_make_app(limits, api_key="secret"))

    response = client.options(
        "/protected",
        headers={
            "Origin": "https://app.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )

    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == "https://app.test"
    assert "authorization" in response.headers["access-control-allow-headers"]
    assert response.headers["access-control-max-age"] == "600"
```

Add to `tests/test_anthropic_server.py`:

```python
def test_anthropic_rate_limit_uses_anthropic_error_shape():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    app = create_app(
        api_key="secret",
        limits=ServerLimits(rate_limit_rpm=1),
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    client = TestClient(app)
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
    }

    assert client.post("/v1/messages", headers={"x-api-key": "secret"}, json=payload).status_code in {200, 503}
    response = client.post("/v1/messages", headers={"x-api-key": "secret"}, json=payload)

    assert response.status_code == 429
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "rate_limited"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_middleware.py tests/test_anthropic_server.py -q
```

Expected failure:

```text
OPTIONS returns 401 or 405
Anthropic rate limit error is OpenAI-shaped
```

- [ ] **Step 3: Implement protocol and preflight helpers**

Add to `src/gemma4_mtp_vllm/server/middleware.py`:

```python
def _protocol_for_path(path: str) -> str:
    return "anthropic" if path.startswith("/v1/messages") else "openai"


def _is_preflight(request: Request) -> bool:
    return (
        request.method == "OPTIONS"
        and request.headers.get("origin") is not None
        and request.headers.get("access-control-request-method") is not None
    )


def _preflight_response(origin: str | None, allowed: set[str], request_id: str) -> Response:
    response = Response(status_code=204)
    _apply_cors(response, origin, allowed)
    response.headers["access-control-max-age"] = "600"
    response.headers["x-request-id"] = request_id
    return response
```

At the start of the middleware after request id:

```python
        if _is_preflight(request):
            return _preflight_response(
                request.headers.get("origin"),
                allowed_origins,
                request_id,
            )
```

Change boundary error creation:

```python
protocol = _protocol_for_path(path)
```

Use `protocol=protocol` for `protocol_error_response()` calls in rate limit and body cap paths.

- [ ] **Step 4: Stamp CORS on early errors**

Change `_stamp()`:

```python
def _stamp(
    response: Response,
    request_id: str,
    *,
    origin: str | None = None,
    allowed_origins: set[str] | None = None,
) -> Response:
    if allowed_origins is not None:
        _apply_cors(response, origin, allowed_origins)
    response.headers["x-request-id"] = request_id
    return response
```

Call `_stamp(error, request_id, origin=request.headers.get("origin"), allowed_origins=allowed_origins)`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_middleware.py tests/test_anthropic_server.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/server/middleware.py tests/test_middleware.py tests/test_anthropic_server.py
git commit -m "fix: handle cors preflight before auth"
```

### Task 7: Health Capability Truthfulness and MTP Observability

**Files:**
- Modify: `src/gemma4_mtp_vllm/backend/vllm_client.py`
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_server_health.py`
- Modify: `tests/test_vllm_client.py`

- [ ] **Step 1: Add tests for truthful health shape and metrics scrape**

Add to `tests/test_vllm_client.py`:

```python
async def test_metrics_text_returns_plain_text():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="vllm:spec_decode_num_draft_tokens_total 4.0\n")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.local",
    ) as http:
        client = VllmClient(http=http, base_url="http://vllm.local")
        assert "draft_tokens" in await client.metrics_text()
```

Update `tests/test_server_health.py`:

```python
def test_health_reports_protocol_specific_streaming_and_backend_batching():
    client = _client()
    response = client.get("/health", headers={"x-api-key": "secret"})
    body = response.json()

    assert body["streaming"] == {
        "openai": "vllm_passthrough_sse",
        "anthropic": "buffered_translation",
    }
    assert body["batching"] == {
        "backend": "vllm_continuous_batching",
        "gateway": "bounded_admission",
    }
    assert "true_token_streaming" not in body
    assert "continuous_batching" not in body
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_vllm_client.py tests/test_server_health.py -q
```

Expected failure:

```text
VllmClient has no metrics_text
health still contains true_token_streaming and continuous_batching
```

- [ ] **Step 3: Add metrics text client method**

Add to `src/gemma4_mtp_vllm/backend/vllm_client.py`:

```python
    async def metrics_text(self) -> str:
        response = await self._http.get("/metrics")
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        return response.text
```

- [ ] **Step 4: Update health response shape**

In `src/gemma4_mtp_vllm/server/app.py`, replace:

```python
            "tools_supported": False,
            "true_token_streaming": True,
            "continuous_batching": True,
            "token_counting": "estimated_word_count",
```

with:

```python
            "tools_supported": False,
            "multimodal_supported": False,
            "streaming": {
                "openai": "vllm_passthrough_sse",
                "anthropic": "buffered_translation",
            },
            "batching": {
                "backend": "vllm_continuous_batching",
                "gateway": "bounded_admission",
            },
            "token_counting": "estimated_word_count",
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_vllm_client.py tests/test_server_health.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/backend/vllm_client.py src/gemma4_mtp_vllm/server/app.py tests/test_vllm_client.py tests/test_server_health.py
git commit -m "fix: report gateway capabilities truthfully"
```

### Task 8: Count Tokens Alias Validation and Release Docs

**Files:**
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_anthropic_server.py`
- Modify: `README.md`
- Modify: `scripts/verify_wheel_freshness.sh`
- Modify: `tests/test_release_scripts.py`

- [ ] **Step 1: Add count tokens alias test**

Add to `tests/test_anthropic_server.py`:

```python
def test_anthropic_count_tokens_rejects_unknown_model():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = _vllm(handler)
    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "secret", "content-type": "application/json"},
        json={
            "model": "not-a-model",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_anthropic_server.py::test_anthropic_count_tokens_rejects_unknown_model -q
```

Expected failure:

```text
status_code == 200
```

- [ ] **Step 3: Add alias check to count_tokens**

In `/v1/messages/count_tokens`, after validation:

```python
        if not _alias_known(payload.get("model"), aliases):
            return protocol_error_response(
                status_code=404,
                code="model_not_found",
                message=f"model {payload.get('model')!r} is not available",
                protocol="anthropic",
            )
```

- [ ] **Step 4: Update release docs**

Update `README.md`:

```markdown
## vLLM Version

The gateway requires `vllm >= 0.21.0,<0.22.0` for Gemma 4 MTP. vLLM
0.21.0 includes Gemma4 MTP support via PR #41745. Older vLLM releases can
treat the Gemma 4 assistant checkpoint as a generic draft model and fail
during initialization.
```

Update the Doctor example version:

```json
{"ok": true, "profile": "safe80", "target_model": "google/gemma-4-31B-it", "drafter": "google/gemma-4-31B-it-assistant", "drafter_configured": "google/gemma-4-31B-it-assistant", "drafter_loaded": "unknown", "num_speculative_tokens": 4, "tensor_parallel_size": 1, "gateway_version": "0.1.0", "required_vllm_min_version": "0.21.0", "vllm": {"status": "ok", "version": "0.21.0"}, "version_ok": true, "target_served": true}
```

Add to Source Archives:

```markdown
Do not publish manually created Finder or desktop zip files. Release source
archives must be created by `scripts/make_source_archive.sh`; the verifier
rejects `.git`, `.venv`, `dist`, `__MACOSX`, `__pycache__`, and build/cache
entries.
```

Update `scripts/verify_wheel_freshness.sh` fake version to:

```python
return httpx.Response(200, json={"status": "ok", "data": [], "version": "0.21.0"})
```

- [ ] **Step 5: Run docs and release tests**

Run:

```bash
python -m pytest tests/test_anthropic_server.py tests/test_release_scripts.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gemma4_mtp_vllm/server/app.py tests/test_anthropic_server.py README.md scripts/verify_wheel_freshness.sh tests/test_release_scripts.py
git commit -m "docs: document vllm mtp release requirements"
```

---

## P1 Correctness and Observability Tasks

### Task 9: Bench Matrix Must Use Startup-Level URLs

**Files:**
- Modify: `src/gemma4_mtp_vllm/cli.py`
- Modify: `tests/test_bench_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing CLI test for mapped MTP URLs**

Add to `tests/test_bench_cli.py`:

```python
def test_bench_matrix_requires_url_per_speculative_depth():
    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--prompt",
            "hi",
            "--mtp-url",
            "http://mtp.local:8001",
            "--baseline-url",
            "http://base.local:8002",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
        ],
    )

    assert result.exit_code == 2
    assert "--mtp-url must be provided as N=URL" in result.output
```

- [ ] **Step 2: Change CLI contract**

Replace `mtp_url: str` in `bench_matrix()` with:

```python
    mtp_url: list[str] = typer.Option([], "--mtp-url"),
```

Add parser:

```python
def _parse_mtp_url_map(values: list[str]) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("--mtp-url must be provided as N=URL")
        key, url = value.split("=", 1)
        try:
            depth = int(key)
        except ValueError as exc:
            raise typer.BadParameter("--mtp-url key must be an integer") from exc
        if depth <= 0 or not url:
            raise typer.BadParameter("--mtp-url must use positive N and non-empty URL")
        parsed[depth] = url
    return parsed
```

Use:

```python
    mtp_urls = _parse_mtp_url_map(mtp_url)
    missing = [n for n in num_speculative_tokens if n not in mtp_urls]
    if missing:
        typer.echo(f"missing --mtp-url for num_speculative_tokens: {missing}", err=True)
        raise typer.Exit(code=2)
```

Call `_single_bench(... mtp_url=mtp_urls[n], ...)` and include `"config_source": "server_startup"` in each result.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_bench_cli.py -q
```

Expected:

```text
passed
```

### Task 10: vLLM MTP Metrics Scrape and Low Acceptance Warning

**Files:**
- Modify: `src/gemma4_mtp_vllm/doctor.py`
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_doctor.py`
- Modify: `tests/test_server_health.py`

- [ ] **Step 1: Add tests for metrics parsing**

Add a helper in `doctor.py` or a new focused function:

```python
def parse_mtp_metrics(text: str, *, model_name: str) -> dict[str, float | bool]:
    ...
```

Tests must cover this input:

```text
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="gemma-4-31b-mtp"} 68.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="gemma-4-31b-mtp"} 35.0
```

Expected output:

```python
{
    "mtp_observed": True,
    "draft_tokens": 68.0,
    "accepted_tokens": 35.0,
    "acceptance_rate": 35.0 / 68.0,
    "low_acceptance_warning": False,
}
```

- [ ] **Step 2: Use threshold warning**

Set low acceptance threshold to `0.05` for alpha:

```python
low_acceptance_warning = mtp_observed and acceptance_rate < 0.05
```

### Task 11: Tokenizer-Exact Count Tokens Spike

**Files:**
- Modify: `src/gemma4_mtp_vllm/backend/vllm_client.py`
- Modify: `src/gemma4_mtp_vllm/server/app.py`
- Modify: `tests/test_anthropic_server.py`

- [ ] **Step 1: Prefer vLLM tokenizer endpoint**

Use vLLM `/tokenize` if available:

```json
{"model": "gemma-4-31b-mtp", "prompt": "hello world"}
```

If vLLM returns `404` or unsupported response shape, keep estimated count and the header:

```text
X-Gemma4-MTP-Token-Counting: estimated_word_count
```

If exact count works, use:

```text
X-Gemma4-MTP-Token-Counting: vllm_tokenizer
```

### Task 12: Non-Streaming Thought Strip Fallback

**Files:**
- Modify: `src/gemma4_mtp_vllm/backend/response_parser.py`
- Modify: `src/gemma4_mtp_vllm/anthropic_adapter.py`
- Modify: `tests/test_vllm_client.py`
- Modify: `tests/test_anthropic_adapter.py`

- [ ] **Step 1: Define exact parser behavior**

Add tests:

```python
def test_visible_text_strips_gemma_thought_channel():
    text = "<|channel|>thought hidden reasoning<|channel|>final visible answer"

    assert visible_text_for_history(text) == "visible answer"
```

```python
def test_visible_text_keeps_plain_text():
    assert visible_text_for_history("plain answer") == "plain answer"
```

- [ ] **Step 2: Apply parser only to non-stream responses**

In `openai_response_to_anthropic()`, pass assistant content through `visible_text_for_history()`.

---

## P2 Agent-Ready Tasks

These are outside public v0.1 release blockers and must not be included in the first remediation branch:

- Tool and function calling through Gemma 4 `--tool-call-parser gemma4`.
- Anthropic true async streaming without buffering the full upstream stream.
- Multimodal request support and visual token budget policy.
- Claude Code and Kilo Code full agent roundtrip compatibility.
- Automated vLLM process orchestration for `bench-matrix`.

---

## Verification Plan

Run local checks after P0:

```bash
python -m pytest -q
python -m pip check
python -m compileall -q src
python -m build --wheel
scripts/verify_wheel_freshness.sh
scripts/make_source_archive.sh dist/Gemma-4-31B-MTP-vllm-src.zip
scripts/verify_source_archive.sh dist/Gemma-4-31B-MTP-vllm-src.zip
```

Run remote real-hardware smoke on the 2x RTX 5090 host after P0:

```bash
PATH=$HOME/vllm-gemma4-mtp-021-venv/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1 \
$HOME/vllm-gemma4-mtp-021-venv/bin/vllm serve \
  $HOME/models/Gemma-4-31B-IT-NVFP4 \
  --served-model-name gemma-4-31b-mtp \
  --host 127.0.0.1 \
  --port 8010 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","model":"/home/homelander/models/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
```

Then verify:

```bash
curl -sS -i http://127.0.0.1:8010/health
curl -sS http://127.0.0.1:8010/version
curl -sS http://127.0.0.1:8010/v1/models
curl -sS http://127.0.0.1:8010/metrics | grep 'spec_decode'
```

Gateway smoke acceptance:

```text
GET /livez -> 200
GET /version -> vllm_version 0.21.0
GET /health -> 200, not 500, with status ready
POST /v1/chat/completions -> 200 and generated text
POST /v1/completions -> 200
POST /v1/messages -> 200
POST /v1/messages/count_tokens -> 200
GET /metrics -> total_requests increments after generation
```

## Self-Review

Spec coverage:

- vLLM minimum and MTP method are covered by Tasks 1 and 8.
- Real vLLM health empty body is covered by Task 2.
- Doctor false negative is covered by Task 3.
- Bounded admission and metrics are covered by Task 5.
- CORS preflight and protocol-aware boundary errors are covered by Task 6.
- Strict typed request validation is covered by Task 4.
- Count token model alias validation is covered by Task 8.
- Health capability truthfulness is covered by Task 7.
- Bench matrix correctness is P1 Task 9 because it does not block the sidecar generation surface.
- Tokenizer exact count and thought stripping are P1 Tasks 11 and 12 because v0.1 already documents estimated token counting and fail-fast tool/multimodal scope.

Placeholder scan:

- No placeholder markers or unspecified validation steps remain.
- Each P0 task has concrete files, tests, commands, expected failures, implementation snippets, and commit command.

Type consistency:

- `RequestValidationError` fields match all app usage.
- `version_at_least(value, minimum)` matches doctor tests.
- `VllmClient.metrics_text()` is used only as text, not JSON.
- `drafter_loaded` is intentionally string `"unknown"` because it is not observable through `/v1/models`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-v0-1-release-blockers.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

