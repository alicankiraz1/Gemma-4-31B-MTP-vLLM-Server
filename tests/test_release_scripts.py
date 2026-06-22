from __future__ import annotations

import os
import stat
import subprocess
import zipfile
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
    assert "git status --porcelain --untracked-files=all" in make_script
    assert "worktree is dirty" in make_script
    assert "git status --porcelain --untracked-files=all" in wheel_script
    assert "worktree is dirty" in wheel_script


def test_make_source_archive_refuses_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "user.name=Release Test",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "LOCAL_ONLY.txt").write_text("not for release\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(Path.cwd() / "scripts/make_source_archive.sh"),
            str(tmp_path / "src.zip"),
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "worktree is dirty" in result.stderr


def test_make_source_archive_uses_committed_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "user.name=Release Test",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    archive = tmp_path / "src.zip"
    result = subprocess.run(
        [str(Path.cwd() / "scripts/make_source_archive.sh"), str(archive)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("README.md").decode("utf-8") == "# committed\n"


def test_readme_documents_vllm_mtp_release_requirement():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "vllm == 0.21.0" in readme
    assert "constraints/vllm-0.21.0-cu130.txt" in readme
    assert "site-packages" in readme
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
    assert '"config_verification": {"status": "partial"' in readme
    assert '"config_matches": false' in readme
    assert '"mtp": {"state": "active"' in readme
    assert '"mtp_observed": true' in readme
    assert "drafter model id is not listed" not in readme


def test_readme_warns_against_manual_source_archives():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Do not publish manually created Finder or desktop zip files" in readme
    assert "scripts/make_source_archive.sh" in readme
    assert "Release artifact scripts refuse a dirty worktree" in readme
    assert "--allow-dirty" in readme
    for path in (
        ".git",
        ".venv",
        ".worktrees",
        "dist",
        "__MACOSX",
        "__pycache__",
        "artifacts",
        "logs",
        "env files",
        "internal/superpowers plan files",
        "local absolute paths",
        "secret-like content",
    ):
        assert path in readme
    assert "build/cache entries" in readme


def test_readme_and_cli_do_not_describe_current_alpha_as_v0_1():
    readme = Path("README.md").read_text(encoding="utf-8")
    cli = Path("src/gemma4_mtp_vllm/cli.py").read_text(encoding="utf-8")
    assert "narrow in v0.1" not in readme
    assert "translation in v0.1" not in readme
    assert "Disabled in v0.1" not in cli


def test_gitignore_covers_env_file_variants():
    for path in (".env", ".env.local", "prod.env", "service.env.local", "nested/service.env.local"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            check=False,
        )
        assert result.returncode == 0, path


def test_verify_source_archive_excludes_forbidden_paths():
    script = Path("scripts/verify_source_archive.sh").read_text(encoding="utf-8")
    for path in (".venv", ".git", "dist", ".pytest_cache", "__pycache__",
                 "__MACOSX", "build", "bench-results", "artifacts", "logs",
                 ".env", "docs/superpowers/plans", "internal/superpowers"):
        assert path in script


def test_verify_source_archive_fails_missing_archive(tmp_path):
    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(tmp_path / "missing.zip")],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "archive not found" in result.stderr


def test_verify_source_archive_fails_corrupt_archive(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_text("not a zip", encoding="utf-8")
    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0


def test_verify_source_archive_accepts_clean_archive(tmp_path):
    archive = tmp_path / "clean.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("README.md", "# ok\n")

    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "archive clean" in result.stdout


def test_verify_source_archive_rejects_env_files(tmp_path):
    for index, filename in enumerate((".env", "prod.env", "service.env.local")):
        archive = tmp_path / f"env-{index}.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(filename, "TOKEN=secret\n")

        result = subprocess.run(
            ["scripts/verify_source_archive.sh", str(archive)],
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0, filename
        assert "forbidden entries" in result.stdout


def test_verify_source_archive_rejects_internal_plan_files(tmp_path):
    archive = tmp_path / "plans.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("docs/superpowers/plans/p0.md", "internal plan\n")

    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "forbidden entries" in result.stdout


def test_verify_source_archive_rejects_absolute_local_paths(tmp_path):
    archive = tmp_path / "local-path.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "README.md",
            "workspace " + "/" + "Users" + "/alicankiraz/private\n",
        )

    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "local or secret content" in result.stdout


def test_verify_source_archive_rejects_secret_like_content(tmp_path):
    archive = tmp_path / "secret.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "README.md",
            "hf_" + "abcdefghijklmnopqrstuvwxyz1234567890\n",
        )

    result = subprocess.run(
        ["scripts/verify_source_archive.sh", str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "local or secret content" in result.stdout
