from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUALITY_BENCHMARK_LANE = "quality_natural_eos"
QUALITY_SEED = 1
QUALITY_MAX_TOKENS = 384


@dataclass(frozen=True)
class QualityTask:
    task_id: str
    category: str
    messages: list[dict[str, str]]
    validator: str
    expected: str | None = None
    schema: dict[str, Any] | None = None
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    files: dict[str, str] | None = None
    tests: dict[str, str] | None = None
    max_tokens: int = QUALITY_MAX_TOKENS


THOUGHT_LEAKAGE_PATTERNS = (
    re.compile(r"(?i)<\s*/?\s*think\s*>"),
    re.compile(r"(?i)\bchain[- ]of[- ]thought\b"),
    re.compile(r"(?i)\bhidden reasoning\b"),
    re.compile(r"(?i)\binternal reasoning\b"),
    re.compile(r"(?im)^\s*(reasoning|analysis)\s*:"),
)

REFUSAL_PATTERNS = (
    re.compile(r"(?i)\bi can(?:not|'t)\b"),
    re.compile(r"(?i)\bi won't\b"),
    re.compile(r"(?i)\bunable to comply\b"),
    re.compile(r"(?i)\bas an ai\b"),
)


def build_quality_request_body(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = QUALITY_MAX_TOKENS,
    seed: int = QUALITY_SEED,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "ignore_eos": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }


def default_quality_suite() -> list[QualityTask]:
    tasks: list[QualityTask] = []
    tasks.extend(_python_unit_tasks())
    tasks.extend(_systems_static_tasks())
    tasks.append(_patch_apply_task())
    tasks.append(_repo_patch_test_task())
    tasks.extend(_json_schema_tasks())
    tasks.extend(_turkish_security_tasks())
    tasks.extend(_reasoning_tasks())
    tasks.extend(_retrieval_tasks())
    tasks.extend(_multi_turn_tasks())
    return tasks


def suite_category_counts(tasks: list[QualityTask]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.category] = counts.get(task.category, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_quality_task(
    task: QualityTask,
    *,
    content: str,
    finish_reason: str | None,
    message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thought_leakage = detect_thought_leakage(content, message=message)
    refusal = detect_refusal(content)
    truncated = finish_reason == "length"
    validation = _validate_task(task, content)
    return {
        "task_id": task.task_id,
        "category": task.category,
        "validator": task.validator,
        "passed": validation["passed"],
        "exact_pass": validation.get("exact_pass"),
        "normalized_pass": validation.get("normalized_pass"),
        "executable_static_pass": validation.get("executable_static_pass"),
        "json_valid": validation.get("json_valid"),
        "schema_valid": validation.get("schema_valid"),
        "patch_apply_success": validation.get("patch_apply_success"),
        "truncated": truncated,
        "refusal": refusal,
        "thought_leakage": thought_leakage,
        "finish_reason": finish_reason,
        "diagnostics": validation.get("diagnostics") or {},
        "output_sha256": _sha256(content),
    }


def quality_report(
    *,
    service_url: str,
    profile: str,
    model: str,
    tasks: list[QualityTask],
    task_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(task_results)
    return {
        "benchmark_lane": QUALITY_BENCHMARK_LANE,
        "benchmark_kind": "single_endpoint_quality",
        "status": "complete",
        "service_url": service_url,
        "profile": profile,
        "model": model,
        "request_contract": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": QUALITY_SEED,
            "ignore_eos": False,
            "min_tokens": "absent",
            "stream": False,
            "max_tokens": "task_realistic",
        },
        "suite_category_counts": suite_category_counts(tasks),
        "summary": {
            "task_count": total,
            "overall_pass_rate": _rate(
                sum(1 for result in task_results if result["passed"]),
                total,
            ),
            "exact_pass_rate": _metric_rate(task_results, "exact_pass"),
            "normalized_pass_rate": _metric_rate(task_results, "normalized_pass"),
            "executable_static_validation_pass_rate": _metric_rate(
                task_results,
                "executable_static_pass",
            ),
            "json_validity_rate": _metric_rate(task_results, "json_valid"),
            "schema_validity_rate": _metric_rate(task_results, "schema_valid"),
            "patch_apply_success_rate": _metric_rate(
                task_results,
                "patch_apply_success",
            ),
            "truncation_rate": _rate(
                sum(1 for result in task_results if result["truncated"]),
                total,
            ),
            "refusal_rate": _rate(
                sum(1 for result in task_results if result["refusal"]),
                total,
            ),
            "thought_leakage_rate": _rate(
                sum(1 for result in task_results if result["thought_leakage"]),
                total,
            ),
        },
        "task_results": task_results,
    }


def detect_thought_leakage(
    content: str,
    *,
    message: dict[str, Any] | None = None,
) -> bool:
    if any(pattern.search(content) for pattern in THOUGHT_LEAKAGE_PATTERNS):
        return True
    if not message:
        return False
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def detect_refusal(content: str) -> bool:
    return any(pattern.search(content) for pattern in REFUSAL_PATTERNS)


def validate_patch_apply(
    patch_text: str,
    *,
    files: dict[str, str],
) -> dict[str, Any]:
    patch = _extract_patch(patch_text)
    if not patch.strip():
        return {"passed": False, "diagnostics": {"error": "patch_missing"}}
    with tempfile.TemporaryDirectory(prefix="gemma4-mtp-quality-patch-") as tmp:
        root = Path(tmp)
        for relative, text in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        result = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=root,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    return {
        "passed": result.returncode == 0,
        "diagnostics": _subprocess_diagnostics(result),
    }


def _validate_task(task: QualityTask, content: str) -> dict[str, Any]:
    if task.validator == "exact":
        passed = content.strip() == (task.expected or "")
        return {"passed": passed, "exact_pass": passed}
    if task.validator == "normalized":
        passed = _normalize(content) == _normalize(task.expected or "")
        return {"passed": passed, "normalized_pass": passed}
    if task.validator == "json_schema":
        parsed = _parse_json_answer(content)
        json_valid = parsed is not None
        schema_valid = json_valid and _matches_schema(parsed, task.schema or {})
        return {
            "passed": bool(schema_valid),
            "json_valid": json_valid,
            "schema_valid": bool(schema_valid),
        }
    if task.validator == "static_contains":
        normalized = _normalize(content)
        required_ok = all(_normalize(term) in normalized for term in task.required_terms)
        forbidden_ok = not any(
            _normalize(term) in normalized for term in task.forbidden_terms
        )
        passed = required_ok and forbidden_ok
        return {"passed": passed, "executable_static_pass": passed}
    if task.validator == "patch_apply":
        result = validate_patch_apply(content, files=task.files or {})
        passed = result["passed"]
        return {
            "passed": passed,
            "executable_static_pass": passed,
            "patch_apply_success": passed,
            "diagnostics": result["diagnostics"],
        }
    if task.validator == "repo_patch_tests":
        result = _validate_repo_patch_tests(
            content,
            files=task.files or {},
            tests=task.tests or {},
        )
        passed = result["passed"]
        return {
            "passed": passed,
            "executable_static_pass": passed,
            "patch_apply_success": result["patch_apply_success"],
            "diagnostics": result["diagnostics"],
        }
    if task.validator == "python_unit":
        result = _validate_python_unit(content, tests=task.tests or {})
        return {
            "passed": result["passed"],
            "executable_static_pass": result["passed"],
            "diagnostics": result["diagnostics"],
        }
    raise ValueError(f"unknown quality validator: {task.validator}")


def _validate_python_unit(content: str, *, tests: dict[str, str]) -> dict[str, Any]:
    code = _extract_code(content, language="python")
    with tempfile.TemporaryDirectory(prefix="gemma4-mtp-quality-python-") as tmp:
        root = Path(tmp)
        (root / "solution.py").write_text(code, encoding="utf-8")
        for relative, text in tests.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
    return {
        "passed": result.returncode == 0,
        "diagnostics": _subprocess_diagnostics(result),
    }


def _validate_repo_patch_tests(
    content: str,
    *,
    files: dict[str, str],
    tests: dict[str, str],
) -> dict[str, Any]:
    patch = _extract_patch(content)
    if not patch.strip():
        return {
            "passed": False,
            "patch_apply_success": False,
            "diagnostics": {"error": "patch_missing"},
        }
    with tempfile.TemporaryDirectory(prefix="gemma4-mtp-quality-repo-") as tmp:
        root = Path(tmp)
        for relative, text in {**files, **tests}.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        apply_result = subprocess.run(
            ["git", "apply", "-"],
            cwd=root,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        if apply_result.returncode != 0:
            return {
                "passed": False,
                "patch_apply_success": False,
                "diagnostics": _subprocess_diagnostics(apply_result),
            }
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
    return {
        "passed": test_result.returncode == 0,
        "patch_apply_success": True,
        "diagnostics": _subprocess_diagnostics(test_result),
    }


def _subprocess_diagnostics(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _parse_json_answer(content: str) -> Any | None:
    text = _strip_code_fence(content.strip(), language="json")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required") or []
        if any(key not in value for key in required):
            return False
        properties = schema.get("properties") or {}
        for key, property_schema in properties.items():
            if key in value and not _matches_schema(value[key], property_schema):
                return False
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            if any(key not in allowed for key in value):
                return False
        return True
    if schema.get("type") == "array":
        return isinstance(value, list)
    if schema.get("type") == "string":
        return isinstance(value, str)
    if schema.get("type") == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema.get("type") == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema.get("type") == "boolean":
        return isinstance(value, bool)
    return True


def _extract_code(content: str, *, language: str) -> str:
    pattern = re.compile(rf"```(?:{language})?\s*(.*?)```", re.DOTALL | re.I)
    match = pattern.search(content)
    return textwrap.dedent(match.group(1) if match else content).strip() + "\n"


def _extract_patch(content: str) -> str:
    text = _strip_code_fence(content, language="diff")
    marker_index = text.find("diff --git ")
    if marker_index >= 0:
        return text[marker_index:]
    return text


def _strip_code_fence(content: str, *, language: str) -> str:
    pattern = re.compile(rf"^```(?:{language})?\s*(.*?)```\s*$", re.DOTALL | re.I)
    match = pattern.match(content)
    return match.group(1) if match else content


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9çğıöşü]+", " ", value.lower()).strip()


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "passed": numerator,
        "total": denominator,
        "rate": (numerator / denominator) if denominator else None,
    }


def _metric_rate(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    applicable = [result for result in results if result.get(key) is not None]
    return _rate(sum(1 for result in applicable if result[key]), len(applicable))


def _sha256(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _python_unit_tasks() -> list[QualityTask]:
    specs = [
        ("two_sum", "def two_sum(nums, target):", "assert two_sum([2,7,11,15],9)==[0,1]"),
        ("slugify", "def slugify(text):", "assert slugify('Hello, World!')=='hello-world'"),
        ("dedupe", "def dedupe_keep_order(items):", "assert dedupe_keep_order([1,2,1,3])==[1,2,3]"),
        ("flatten", "def flatten_once(items):", "assert flatten_once([[1,2],[3]])==[1,2,3]"),
        ("clamp", "def clamp(value, low, high):", "assert clamp(9,1,5)==5"),
        ("parse_bool", "def parse_bool(text):", "assert parse_bool('yes') is True and parse_bool('0') is False"),
        ("chunked", "def chunked(items, size):", "assert chunked([1,2,3],2)==[[1,2],[3]]"),
        ("is_palindrome", "def is_palindrome(text):", "assert is_palindrome('A man, a plan, a canal: Panama')"),
        ("merge_counts", "def merge_counts(left, right):", "assert merge_counts({'a':1},{'a':2,'b':1})=={'a':3,'b':1}"),
        ("safe_div", "def safe_div(a, b):", "assert safe_div(6,3)==2 and safe_div(1,0) is None"),
    ]
    tasks = []
    for index, (name, signature, assertion) in enumerate(specs, start=1):
        tasks.append(
            QualityTask(
                task_id=f"py_unit_{index:02d}_{name}",
                category="coding_python_unit",
                validator="python_unit",
                messages=[
                    _system_message(),
                    {
                        "role": "user",
                        "content": (
                            f"Task {name}: return only Python code defining "
                            f"`{signature}`. No prose."
                        ),
                    },
                ],
                tests={"test_solution.py": f"from solution import *\n{assertion}\n"},
            )
        )
    return tasks


def _systems_static_tasks() -> list[QualityTask]:
    specs = [
        ("rust_result", ("Result<", "map_err"), ("unwrap()",)),
        ("rust_no_unsafe", ("unsafe", "forbid"), ()),
        ("c_bounds", ("size_t", "len"), ("strcpy",)),
        ("linux_epoll", ("epoll", "nonblocking"), ()),
        ("atomic_ordering", ("Acquire", "Release"), ("SeqCst everywhere",)),
    ]
    return [
        QualityTask(
            task_id=f"systems_static_{index:02d}_{name}",
            category="coding_systems_static",
            validator="static_contains",
            messages=[
                _system_message(),
                {
                    "role": "user",
                    "content": (
                        f"Task {name}: give a compact Rust/systems answer "
                        "covering the requested validation points."
                    ),
                },
            ],
            required_terms=required,
            forbidden_terms=forbidden,
        )
        for index, (name, required, forbidden) in enumerate(specs, start=1)
    ]


def _patch_apply_task() -> QualityTask:
    return QualityTask(
        task_id="patch_apply_01_config_default",
        category="patch_apply",
        validator="patch_apply",
        messages=[
            _system_message(),
            {
                "role": "user",
                "content": "Return a unified git patch changing TIMEOUT from 30 to 45 in config.py.",
            },
        ],
        files={"config.py": "TIMEOUT = 30\n"},
    )


def _repo_patch_test_task() -> QualityTask:
    return QualityTask(
        task_id="bug_fix_tests_01_divide_zero",
        category="bug_fix_tests",
        validator="repo_patch_tests",
        messages=[
            _system_message(),
            {
                "role": "user",
                "content": (
                    "Return a git patch fixing calc.py so divide(1, 0) returns None "
                    "while existing division still works."
                ),
            },
        ],
        files={"calc.py": "def divide(a, b):\n    return a / b\n"},
        tests={
            "test_calc.py": (
                "from calc import divide\n\n"
                "def test_divide_ok():\n    assert divide(6, 3) == 2\n\n"
                "def test_divide_zero():\n    assert divide(1, 0) is None\n"
            )
        },
    )


def _json_schema_tasks() -> list[QualityTask]:
    schema = {
        "type": "object",
        "required": ["status", "score", "tags"],
        "properties": {
            "status": {"type": "string"},
            "score": {"type": "integer"},
            "tags": {"type": "array"},
        },
        "additionalProperties": False,
    }
    return [
        QualityTask(
            task_id=f"json_schema_{index:02d}",
            category="structured_json",
            validator="json_schema",
            messages=[
                _system_message(),
                {
                    "role": "user",
                    "content": (
                        "Return only JSON with status:string, score:integer, "
                        f"tags:array for item {index}."
                    ),
                },
            ],
            schema=schema,
        )
        for index in range(1, 11)
    ]


def _turkish_security_tasks() -> list[QualityTask]:
    return [
        QualityTask(
            task_id=f"turkish_security_{index:02d}",
            category="turkish_technical_security",
            validator="static_contains",
            messages=[
                _system_message(),
                {
                    "role": "user",
                    "content": (
                        "Turkce yanitla: bir teknik guvenlik incelemesinde "
                        f"risk, etki ve onlem maddelerini kisaca yaz. Senaryo {index}."
                    ),
                },
            ],
            required_terms=("risk", "etki", "onlem"),
        )
        for index in range(1, 11)
    ]


def _reasoning_tasks() -> list[QualityTask]:
    answers = [
        "17",
        "blue",
        "42",
        "B",
        "13",
        "north",
        "7",
        "valid",
        "3",
        "false",
    ]
    return [
        QualityTask(
            task_id=f"reasoning_exact_{index:02d}",
            category="deterministic_reasoning",
            validator="exact",
            expected=answer,
            messages=[
                _system_message(),
                {
                    "role": "user",
                    "content": f"Answer only the final value for deterministic puzzle {index}: {answer}",
                },
            ],
        )
        for index, answer in enumerate(answers, start=1)
    ]


def _retrieval_tasks() -> list[QualityTask]:
    return [
        QualityTask(
            task_id=f"retrieval_context_{index:02d}",
            category="retrieval_context_grounded",
            validator="normalized",
            expected=f"fact-{index}",
            messages=[
                _system_message(),
                {
                    "role": "user",
                    "content": (
                        f"Context: alpha=fact-{index}; beta=ignore. "
                        "Question: what is alpha? Answer only the cited value."
                    ),
                },
            ],
        )
        for index in range(1, 11)
    ]


def _multi_turn_tasks() -> list[QualityTask]:
    return [
        QualityTask(
            task_id=f"multi_turn_history_{index:02d}",
            category="multi_turn_history_sensitive",
            validator="normalized",
            expected=f"ticket-{index}",
            messages=[
                _system_message(),
                {"role": "user", "content": f"Remember ticket id ticket-{index}."},
                {"role": "assistant", "content": "Stored."},
                {"role": "user", "content": "Return only the ticket id I gave you."},
            ],
        )
        for index in range(1, 6)
    ]


def _system_message() -> dict[str, str]:
    return {
        "role": "system",
        "content": "Be deterministic. Return only the requested final artifact.",
    }
