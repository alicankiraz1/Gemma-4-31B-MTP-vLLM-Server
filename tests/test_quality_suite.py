from __future__ import annotations

import subprocess

from gemma4_mtp_vllm.quality_suite import (
    QualityTask,
    build_quality_request_body,
    default_quality_suite,
    detect_thought_leakage,
    evaluate_quality_task,
    suite_category_counts,
    validate_patch_apply,
)


def test_quality_request_body_is_natural_eos_without_min_tokens():
    body = build_quality_request_body(
        model="gemma-4-31b-mtp",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=256,
    )

    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 1
    assert body["ignore_eos"] is False
    assert body["stream"] is False
    assert body["max_tokens"] == 256
    assert "min_tokens" not in body


def test_default_quality_suite_category_counts():
    counts = suite_category_counts(default_quality_suite())

    assert counts == {
        "bug_fix_tests": 1,
        "coding_python_unit": 10,
        "coding_systems_static": 5,
        "deterministic_reasoning": 10,
        "multi_turn_history_sensitive": 5,
        "patch_apply": 1,
        "retrieval_context_grounded": 10,
        "structured_json": 10,
        "turkish_technical_security": 10,
    }


def test_json_schema_validation_tracks_json_and_schema_validity():
    task = QualityTask(
        task_id="json_test",
        category="structured_json",
        validator="json_schema",
        schema={
            "type": "object",
            "required": ["status", "score"],
            "properties": {
                "status": {"type": "string"},
                "score": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        messages=[],
    )

    passed = evaluate_quality_task(
        task,
        content='{"status":"ok","score":3}',
        finish_reason="stop",
    )
    wrong_schema = evaluate_quality_task(
        task,
        content='{"status":"ok","score":"3"}',
        finish_reason="stop",
    )
    invalid_json = evaluate_quality_task(
        task,
        content="{not-json}",
        finish_reason="stop",
    )

    assert passed["passed"] is True
    assert passed["json_valid"] is True
    assert passed["schema_valid"] is True
    assert wrong_schema["json_valid"] is True
    assert wrong_schema["schema_valid"] is False
    assert invalid_json["json_valid"] is False
    assert invalid_json["schema_valid"] is False


def test_thought_leakage_detection_checks_content_and_reasoning_fields():
    assert detect_thought_leakage("<think>hidden</think> final") is True
    assert detect_thought_leakage("final", message={"reasoning_content": "hidden"})
    assert detect_thought_leakage("plain final", message={"content": "plain final"}) is False


def test_patch_apply_validation_accepts_git_patch():
    patch = """diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1 +1 @@
-TIMEOUT = 30
+TIMEOUT = 45
"""

    result = validate_patch_apply(patch, files={"config.py": "TIMEOUT = 30\n"})

    assert result["passed"] is True


def test_patch_apply_validation_rejects_non_applicable_patch():
    patch = """diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1 +1 @@
-MISSING = 30
+TIMEOUT = 45
"""

    result = validate_patch_apply(patch, files={"config.py": "TIMEOUT = 30\n"})

    assert result["passed"] is False


def test_patch_apply_validation_rejects_unexpected_paths_before_git_apply(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("git apply should not run for unexpected paths")

    monkeypatch.setattr("gemma4_mtp_vllm.quality_suite.subprocess.run", fail_run)
    patch = """diff --git a/secrets.py b/secrets.py
--- a/secrets.py
+++ b/secrets.py
@@ -1 +1 @@
-TOKEN = "old"
+TOKEN = "new"
"""

    result = validate_patch_apply(patch, files={"config.py": "TIMEOUT = 30\n"})

    assert result["passed"] is False
    assert result["diagnostics"]["error"] == "unexpected_patch_paths"
    assert result["diagnostics"]["unexpected_paths"] == ["secrets.py"]


def test_patch_apply_validation_rejects_git_metadata_before_git_apply(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("git apply should not run for unsupported metadata")

    monkeypatch.setattr("gemma4_mtp_vllm.quality_suite.subprocess.run", fail_run)
    symlink_patch = """diff --git a/calc.py b/calc.py
new file mode 120000
--- /dev/null
+++ b/calc.py
@@ -0,0 +1 @@
+/etc/passwd
"""
    copy_patch = """diff --git a/calc.py b/calc.py
copy to conftest.py
--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""

    symlink_result = validate_patch_apply(
        symlink_patch,
        files={"calc.py": "VALUE = 1\n"},
    )
    copy_result = validate_patch_apply(copy_patch, files={"calc.py": "VALUE = 1\n"})

    assert symlink_result["passed"] is False
    assert symlink_result["diagnostics"]["error"] == "unsupported_patch_metadata"
    assert copy_result["passed"] is False
    assert copy_result["diagnostics"]["error"] == "unsupported_patch_metadata"


def test_python_unit_validation_accepts_safe_solution():
    task = QualityTask(
        task_id="py_unit_safe",
        category="coding_python_unit",
        validator="python_unit",
        messages=[],
        tests={"test_solution.py": "from solution import double\n\ndef test_double():\n    assert double(4) == 8\n"},
    )

    result = evaluate_quality_task(
        task,
        content="def double(value):\n    return value * 2\n",
        finish_reason="stop",
    )

    assert result["passed"] is True
    assert result["executable_static_pass"] is True
    assert "stdout_tail" not in result["diagnostics"]
    assert "stderr_tail" not in result["diagnostics"]


def test_python_unit_validation_rejects_unsafe_code_before_pytest(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("pytest should not run for unsafe model code")

    monkeypatch.setattr("gemma4_mtp_vllm.quality_suite.subprocess.run", fail_run)
    task = QualityTask(
        task_id="py_unit_unsafe",
        category="coding_python_unit",
        validator="python_unit",
        messages=[],
        tests={"test_solution.py": "from solution import load\n"},
    )

    result = evaluate_quality_task(
        task,
        content="import os\n\ndef load():\n    return open('/etc/passwd').read()\n",
        finish_reason="stop",
    )

    assert result["passed"] is False
    assert result["diagnostics"]["error"] == "unsafe_python_code"
    assert "import_not_allowed" in result["diagnostics"]["violations"]
    assert "call_not_allowed:open" in result["diagnostics"]["violations"]


def test_python_unit_validation_rejects_attrgetter_escape_before_runner(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("runner should not run for AST escape attempts")

    monkeypatch.setattr("gemma4_mtp_vllm.quality_suite.subprocess.run", fail_run)
    task = QualityTask(
        task_id="py_unit_attrgetter",
        category="coding_python_unit",
        validator="python_unit",
        messages=[],
        tests={"test_solution.py": "from solution import load\n"},
    )

    result = evaluate_quality_task(
        task,
        content=(
            "import operator\n\n"
            "def load():\n"
            "    return operator.attrgetter('__globals__')(load)\n"
        ),
        finish_reason="stop",
    )

    assert result["passed"] is False
    assert result["diagnostics"]["error"] == "unsafe_python_code"
    assert "import_not_allowed" in result["diagnostics"]["violations"]
    assert "string_token_not_allowed:__" in result["diagnostics"]["violations"]


def test_python_unit_validation_timeout_is_task_failure_without_raw_output(monkeypatch):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="RAW MODEL OUTPUT",
            stderr="RAW STDERR",
        )

    monkeypatch.setattr("gemma4_mtp_vllm.quality_suite.subprocess.run", timeout_run)
    task = QualityTask(
        task_id="py_unit_timeout",
        category="coding_python_unit",
        validator="python_unit",
        messages=[],
        tests={"test_solution.py": "from solution import identity\n"},
    )

    result = evaluate_quality_task(
        task,
        content="def identity(value):\n    return value\n",
        finish_reason="stop",
    )

    assert result["passed"] is False
    assert result["diagnostics"]["error"] == "timeout"
    assert "RAW MODEL OUTPUT" not in str(result["diagnostics"])
    assert "stdout_sha256" in result["diagnostics"]
