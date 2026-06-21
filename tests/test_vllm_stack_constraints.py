from __future__ import annotations

from pathlib import Path


def test_vllm_stack_constraints_pin_compatible_http_stack():
    constraints = Path("constraints/vllm-0.21.0-cu130.txt").read_text(
        encoding="utf-8"
    )

    assert "vllm==0.21.0" in constraints
    assert "fastapi==0.136.3" in constraints
    assert "prometheus-fastapi-instrumentator==7.1.0" in constraints


def test_http_stack_smoke_script_exercises_instrumentator():
    script = Path("scripts/smoke_vllm_http_stack.py").read_text(encoding="utf-8")

    assert "prometheus_fastapi_instrumentator" in script
    assert "Instrumentator().instrument(app).expose(app)" in script
    assert 'get("/health")' in script
