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
    assert '"version": "0.21.0"' in script
    assert "--allow-dirty" in script
    assert "worktree is dirty" in script


def test_release_scripts_refuse_dirty_worktrees():
    make_script = Path("scripts/make_source_archive.sh").read_text(encoding="utf-8")
    wheel_script = Path("scripts/verify_wheel_freshness.sh").read_text(encoding="utf-8")
    for script in (make_script, wheel_script):
        assert "git diff --quiet" in script
        assert "git diff --cached --quiet" in script
        assert "worktree is dirty" in script


def test_readme_documents_vllm_mtp_release_requirement():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "vllm >= 0.21.0,<0.22.0" in readme
    assert "Older vLLM releases" in readme
    assert "assistant checkpoint" in readme


def test_readme_doctor_example_matches_current_shape():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert '"required_vllm_min_version": "0.21.0"' in readme
    assert '"vllm": {"status": "ok", "version": "0.21.0"}' in readme
    assert '"drafter_configured": "google/gemma-4-31B-it-assistant"' in readme
    assert '"drafter_loaded": "unknown"' in readme
    assert '"version_ok": true' in readme
    assert '"target_served": true' in readme
    assert "drafter model id is not listed" not in readme


def test_readme_warns_against_manual_source_archives():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Do not publish manually created Finder or desktop zip files" in readme
    assert "scripts/make_source_archive.sh" in readme
    assert "Release artifact scripts refuse a dirty worktree" in readme
    assert "--allow-dirty" in readme
    for path in (".git", ".venv", "dist", "__MACOSX", "__pycache__"):
        assert path in readme
    assert "build/cache entries" in readme


def test_verify_source_archive_excludes_forbidden_paths():
    script = Path("scripts/verify_source_archive.sh").read_text(encoding="utf-8")
    for path in (".venv", ".git", "dist", ".pytest_cache", "__pycache__",
                 "__MACOSX", "build", "bench-results"):
        assert path in script
