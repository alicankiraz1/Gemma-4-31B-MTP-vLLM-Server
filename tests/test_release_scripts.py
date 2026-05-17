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
