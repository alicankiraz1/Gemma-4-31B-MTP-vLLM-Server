import json
import shlex
import subprocess

from gemma4_mtp_vllm.launch import (
    _git_dirty,
    build_launch_manifest,
    build_vllm_serve_args,
    resolve_vllm_executable,
    write_launch_manifest,
)
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


def _smoke_profile():
    return resolve_profile("tp2_2x32_smoke", load_profiles())


def test_build_args_includes_target_and_speculative_config():
    profile = _profile()
    args = build_vllm_serve_args(profile=profile, host="127.0.0.1", port=8000)
    assert args[0] == "vllm"
    assert "serve" in args
    assert "google/gemma-4-31B-it" in args
    spec_idx = args.index("--speculative-config")
    spec = json.loads(args[spec_idx + 1])
    assert spec == {
        "method": "mtp",
        "model": profile.drafter,
        "num_speculative_tokens": profile.num_speculative_tokens,
    }
    assert "--tensor-parallel-size" in args
    assert "--max-model-len" in args
    assert "--gpu-memory-utilization" in args
    assert "--cpu-offload-gb" in args
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "8000" in args


def test_build_args_enable_gemma4_reasoning_parser():
    args = build_vllm_serve_args(profile=_profile())
    parser_idx = args.index("--reasoning-parser")
    assert args[parser_idx + 1] == "gemma4"


def test_build_args_can_set_served_model_name():
    args = build_vllm_serve_args(
        profile=_profile(),
        served_model_name="gemma-4-31b-mtp",
    )
    served_idx = args.index("--served-model-name")
    assert args[served_idx + 1] == "gemma-4-31b-mtp"


def test_build_args_round_trip_through_shell_join():
    args = build_vllm_serve_args(profile=_profile())
    assert shlex.split(shlex.join(args)) == args


def test_build_args_can_disable_mtp_for_baseline():
    args = build_vllm_serve_args(
        profile=_profile(),
        host="127.0.0.1",
        port=8000,
        enable_mtp=False,
    )
    assert "--speculative-config" not in args


def test_build_args_reproduce_homelander_smoke_profile():
    args = build_vllm_serve_args(
        profile=_smoke_profile(),
        host="127.0.0.1",
        port=8012,
        served_model_name="gemma-4-31b-mtp",
    )

    assert args[0:3] == ["vllm", "serve", "google/gemma-4-31B-it"]
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--max-model-len") + 1] == "2048"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
    assert args[args.index("--cpu-offload-gb") + 1] == "8"
    assert args[args.index("--max-num-seqs") + 1] == "1"
    assert args[args.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--enforce-eager" in args
    assert "--language-model-only" not in args


def test_build_args_include_gpu_only_quantization_profile():
    args = build_vllm_serve_args(profile=resolve_profile("tp2_2x32_fp8_gpuonly", load_profiles()))

    assert args[args.index("--cpu-offload-gb") + 1] == "0"
    assert args[args.index("--quantization") + 1] == "fp8"


def test_build_launch_manifest_includes_runtime_fingerprint():
    profile = _smoke_profile()
    args = build_vllm_serve_args(profile=profile, served_model_name="gemma-4-31b-mtp")

    manifest = build_launch_manifest(
        profile=profile,
        argv=args,
        enable_mtp=True,
        served_model_name="gemma-4-31b-mtp",
    )

    assert manifest["profile"] == "tp2_2x32_smoke"
    assert manifest["served_model_name"] == "gemma-4-31b-mtp"
    assert manifest["enable_mtp"] is True
    assert manifest["argv"] == args
    assert manifest["pid"] > 0
    assert "timestamp" in manifest
    assert "git_sha" in manifest
    assert "package_versions" in manifest
    assert "gemma4-mtp-vllm" in manifest["package_versions"]


def test_launch_manifest_git_dirty_includes_untracked_files(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=launch@example.invalid",
            "-c",
            "user.name=Launch Test",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "LOCAL_ONLY.txt").write_text("not in manifest baseline\n", encoding="utf-8")

    monkeypatch.chdir(repo)

    assert _git_dirty() is True


def test_write_launch_manifest_creates_parent_directory(tmp_path):
    profile = _smoke_profile()
    args = build_vllm_serve_args(profile=profile)
    path = tmp_path / "logs" / "manifest.json"

    write_launch_manifest(
        path=path,
        profile=profile,
        argv=args,
        enable_mtp=True,
        served_model_name="gemma-4-31b-mtp",
    )

    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["profile"] == "tp2_2x32_smoke"
    assert body["argv"] == args


def test_resolve_vllm_executable_falls_back_to_active_env_bin(monkeypatch, tmp_path):
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    vllm = bin_dir / "vllm"
    python.write_text("", encoding="utf-8")
    vllm.write_text("", encoding="utf-8")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("gemma4_mtp_vllm.launch.sys.executable", str(python))

    assert resolve_vllm_executable("vllm") == str(vllm)
