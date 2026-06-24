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
    assert "__version__" in script
    assert '"0.2.0a1"' in script
    assert "raise SystemExit" in script
    assert "installed wheel version mismatch" in script
    assert "/livez smoke failed" in script
    assert "/health smoke failed" in script
    assert "/health smoke missing Gemma model evidence" in script
    assert "assert livez" not in script
    assert "assert health" not in script
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


def test_readme_scopes_public_benchmark_claims_to_immutable_result():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Benchmark ID: `homelander-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0`" in readme
    assert "not a universal Gemma 4 MTP performance claim" in readme
    assert "2x NVIDIA GeForce RTX 5090, 32 GB each" in readme
    assert "`profile`: `tp2_2x32_fp8_gpuonly`" in readme
    assert "`cpu_offload_gb`: `0`" in readme
    assert "`quantization`: `fp8`" in readme
    assert "`max_model_len`: `2048`" in readme
    assert "`max_num_seqs`: `1`" in readme
    assert "`enforce_eager`: `true`" in readme
    assert "`num_speculative_tokens`: `4`" in readme
    assert "direct vLLM endpoint A/B" in readme
    assert "gateway-overhead test" in readme
    assert "MTP vs no-MTP speedup test" in readme
    assert "BF16 CPU-offload smoke" in readme
    assert "FP8 GPU-only result" in readme
    assert "`e2e_output_tokens_per_second`" in readme
    assert "`--artifact-id homelander-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0`" in readme
    for value in ("3.46x", "3.89x", "3.99x", "47.79", "54.02", "55.03"):
        assert value in readme


def test_readme_removes_stale_unscoped_benchmark_numbers():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert not Path("assets/gemma4-mtp-benchmark-card.png").exists()
    for stale in (
        "2.12x_average",
        "132.56_tok",
        "62.74",
        "136.27",
        "62.96",
        "130.71",
        "62.70",
        "132.56",
        "2.17x",
        "2.08x",
        "2.11x",
    ):
        assert stale not in readme


def test_benchmark_record_documents_reproduction_without_local_paths():
    record = Path(
        "docs/benchmarks/"
        "homelander-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0.md"
    ).read_text(encoding="utf-8")
    assert "local hardware/configuration result" in record
    assert "direct vLLM endpoint A/B" in record
    assert "`tp2_2x32_fp8_gpuonly`" in record
    assert "CPU offload: 0 GB" in record
    assert "Quantization: fp8" in record
    assert "Speculative depth: 4" in record
    assert "`e2e_output_tokens_per_second`" in record
    assert "vllm-mtp bench" in record
    assert "--artifact-id homelander-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0" in record
    for value in ("3.46x", "3.89x", "3.99x", "47.79", "54.02", "55.03"):
        assert value in record
    for forbidden in (
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "Ali" + ".63",
        "192.168" + ".",
        "10.0.42" + ".",
    ):
        assert forbidden not in record


def test_p1_001_experiment_runbook_keeps_eager_ab_scoped():
    plan = Path("docs/plans/p1-001-cuda-graph-eager-ab.md").read_text(
        encoding="utf-8"
    )
    runbook = Path("docs/experiments/p1-001-cuda-graph-eager-ab.md").read_text(
        encoding="utf-8"
    )
    combined = plan + runbook
    assert "`tp2_2x32_fp8_gpuonly`" in combined
    assert "`tp2_2x32_fp8_gpuonly_cuda_graph`" in combined
    assert "`enforce_eager: true`" in combined
    assert "`enforce_eager: false`" in combined
    assert "differ only by `enforce_eager`" in combined
    assert "Operator Stop Gate" in combined
    assert "Stop here until the operator approves" in combined
    assert "Do not run it" in combined
    assert "beside the validated live backend" in combined
    assert "Control Stop Gate" in combined
    assert "Stop the control backend before starting the candidate" in combined
    assert "Rollback Gate" in combined
    assert "startup time" in combined
    assert "peak GPU memory" in combined
    assert "TTFT" in combined
    assert "TPOT" in combined
    assert "`e2e_output_tokens_per_second`" in combined
    assert "vllm-mtp bench-single" in combined
    assert "--json-output bench-results/p1-001/eager-true.json" in combined
    assert "--json-output bench-results/p1-001/eager-false.json" in combined
    assert "vllm-mtp bench-compare" in combined
    assert "--control-json bench-results/p1-001/eager-true.json" in combined
    assert "--candidate-json bench-results/p1-001/eager-false.json" in combined
    assert "--control-startup-seconds" in combined
    assert "--candidate-startup-seconds" in combined
    assert "--control-peak-gpu-memory-mib" in combined
    assert "--candidate-peak-gpu-memory-mib" in combined
    assert "--soak-passed" in combined
    assert "--soak-seconds 3600" in combined
    assert "--soak-error-count 0" in combined
    assert "--no-oom" in combined
    assert "--json-output bench-results/p1-001/eager-ab-recommendation.json" in combined
    assert "`change_default_profile: false`" in combined
    assert "wrong profiles" in combined
    assert "differ by more than `enforce_eager`" in combined
    assert "missing 64/256/512/1024 output-token targets" in combined
    assert "mismatched request bodies" in combined
    assert "missing TTFT/TPOT evidence" in combined
    assert "incomplete per-GPU memory samples" in combined
    assert "non-zero soak errors" in combined
    assert "token ids" in combined
    assert "`parity_ready: true`" in combined
    assert "`tokenization_status: unavailable`" in combined
    assert "one-hour soak" in combined
    assert "deterministic parity" in combined
    assert runbook.index("Operator Stop Gate") < runbook.index("## Control Run")
    assert runbook.index("Control Stop Gate") < runbook.index("## Candidate Run")
    assert runbook.index("## Candidate Run") < runbook.index("Recommendation Compare")
    assert runbook.index("Recommendation Compare") < runbook.index("Rollback Gate")


def test_p1_001r_runbook_documents_repaired_2x2_gates():
    runbook = Path("docs/experiments/p1-001r-cuda-graph-2x2-maintenance.md").read_text(
        encoding="utf-8"
    )
    assert "operator explicitly approves the maintenance window" in runbook
    assert "`codex/p0-008-p1-001r-code-gate`" in runbook
    assert "record `git rev-parse HEAD`" in runbook
    assert "`01dc54a93cc46d2513b40acd4a268b22d0c1f6bf`" in runbook
    assert "`127.0.0.1:8012`" in runbook
    assert "`127.0.0.1:18082`" in runbook
    assert "Live default profile must not change" in runbook
    assert "`change_default_profile` must remain `false`" in runbook
    for config in ("A", "B", "C", "D"):
        assert f"| {config} |" in runbook
    for port in ("8111", "8112", "8113", "8114"):
        assert port in runbook
    assert "`--enforce-eager`" in runbook
    assert "`--speculative-config`" in runbook
    assert "return_token_ids: true" in runbook
    assert "same-mode MTP parity: A vs B and C vs D" in runbook
    assert "cross-mode parity: B vs D as diagnostic only" in runbook
    assert "Do not treat eager-vs-graph raw-token inequality alone as `do_not_adopt`" in runbook
    assert "acceptance-rate margin: `-0.01`" in runbook
    assert "mean-acceptance-length margin: `-0.05`" in runbook
    assert "new 10-minute D sanity soak" in runbook
    assert "only after `bench-2x2-compare` reports" in runbook
    assert "passed same-mode A-vs-B and C-vs-D gates" in runbook
    assert "--same-mode-mtp-parity passed" in runbook
    assert "--final-answer-quality passed" in runbook
    assert "doctor, `/health`, OpenAI chat, OpenAI streaming" in runbook
    assert runbook.index("Pre-Stop Gate") < runbook.index("Stop Gate")
    assert runbook.index("Stop Gate") < runbook.index("Matrix")
    assert runbook.index("Matrix") < runbook.index("Candidate Sanity Soak")
    assert runbook.index("Candidate Sanity Soak") < runbook.index("Rollback Gate")


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
