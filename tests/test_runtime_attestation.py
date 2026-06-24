from __future__ import annotations

from dataclasses import replace

from gemma4_mtp_vllm import REQUIRED_VLLM_MIN_VERSION
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile
from gemma4_mtp_vllm.runtime_config import (
    build_config_verification,
    config_matches,
    desired_config,
    merge_observed_config,
    observed_config_from_active_manifest,
    observed_config_from_runtime_evidence,
    observed_config_from_models,
    observed_config_from_metrics,
    observed_config_from_process_argv,
    redact_argv,
    read_text_tail,
)


def _fp8_profile():
    return resolve_profile("tp2_2x32_fp8_gpuonly", load_profiles())


def _fp8_argv(cpu_offload_gb: str = "0") -> list[str]:
    return [
        "vllm",
        "serve",
        "google/gemma-4-31B-it",
        "--port",
        "8000",
        "--served-model-name",
        "gemma-4-31b-mtp",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.95",
        "--cpu-offload-gb",
        cpu_offload_gb,
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
        "--quantization",
        "fp8",
        "--speculative-config",
        '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
    ]


def _models_body() -> dict:
    return {"data": [{"id": "gemma-4-31b-mtp", "max_model_len": 2048}]}


def test_attestation_can_be_fully_verified_from_runtime_sources():
    desired = desired_config(_fp8_profile(), served_model_name="gemma-4-31b-mtp")
    observed = observed_config_from_models(
        _models_body(),
        target_model=_fp8_profile().target,
        served_model_name="gemma-4-31b-mtp",
    )
    observed = merge_observed_config(
        observed,
        observed_config_from_process_argv(_fp8_argv()),
        {
            "vllm_version": "0.21.0",
            "torch_version": "2.11.0+cu130",
            "cuda_version": "13.0",
            "_sources": {
                "vllm_version": "vllm_version_api",
                "torch_version": "runtime_manifest",
                "cuda_version": "nvidia_smi",
            },
        },
    )

    verification = build_config_verification(desired, observed)

    assert verification["status"] == "verified"
    assert config_matches(desired, observed) is True
    assert verification["fields"]["cpu_offload_gb"]["status"] == "verified"
    assert verification["fields"]["quantization"]["source"] == "process_cmdline"


def test_cpu_offload_mismatch_is_not_verified():
    desired = desired_config(_fp8_profile(), served_model_name="gemma-4-31b-mtp")
    observed = observed_config_from_process_argv(_fp8_argv(cpu_offload_gb="8"))

    verification = build_config_verification(desired, observed)

    assert verification["status"] == "mismatch"
    assert verification["fields"]["cpu_offload_gb"]["observed"] == 8.0
    assert verification["fields"]["cpu_offload_gb"]["status"] == "mismatch"
    assert config_matches(desired, observed) is False


def test_profile_only_quantization_is_unknown_not_verified():
    desired = desired_config(_fp8_profile(), served_model_name="gemma-4-31b-mtp")
    observed = observed_config_from_models(
        _models_body(),
        target_model=_fp8_profile().target,
        served_model_name="gemma-4-31b-mtp",
    )

    verification = build_config_verification(desired, observed)

    assert verification["status"] == "partial"
    assert verification["fields"]["quantization"] == {
        "desired": "fp8",
        "observed": None,
        "status": "unknown",
        "source": "unknown",
        "reason": "not observed from runtime",
    }
    assert config_matches(desired, observed) is False


def test_desired_config_redacts_private_profile_paths():
    profile = replace(
        _fp8_profile(),
        target="/" + "home" + "/private-user/private-target",
        drafter="/" + "Users" + "/private-user/private-drafter",
    )

    desired = desired_config(profile, served_model_name="gemma-4-31b-mtp")
    verification = build_config_verification(
        desired,
        {
            "target_model": "REDACTED_PATH",
            "_sources": {"target_model": "process_cmdline"},
        },
    )

    assert desired["target_model"] == "REDACTED_PATH"
    assert desired["drafter"] == "REDACTED_PATH"
    assert verification["fields"]["target_model"]["status"] == "unknown"


def test_models_api_redacts_private_model_ids():
    target_path = "/" + "home" + "/private-user/private-target"
    observed = observed_config_from_models(
        {"data": [{"id": target_path, "max_model_len": 2048}]},
        target_model=target_path,
        served_model_name=None,
    )

    assert observed["models"] == ["REDACTED_PATH"]
    assert observed["target_model"] == "REDACTED_PATH"
    assert observed["target_served"] is True


def test_merge_does_not_erase_runtime_observation_with_unknown_model_value():
    observed = merge_observed_config(
        observed_config_from_process_argv(_fp8_argv()),
        observed_config_from_models(
            _models_body(),
            target_model=_fp8_profile().target,
            served_model_name="gemma-4-31b-mtp",
        ),
    )

    assert observed["target_model"] == "google/gemma-4-31B-it"
    assert observed["_sources"]["target_model"] == "process_cmdline"


def test_unknown_when_no_runtime_observations_exist():
    verification = build_config_verification(
        desired_config(_fp8_profile(), served_model_name="gemma-4-31b-mtp"),
        {},
    )

    assert verification["status"] == "unknown"
    assert verification["fields"]["target_served"]["status"] == "unknown"


def test_active_manifest_requires_matching_pid_and_redacts_argv():
    manifest = {
        "pid": 123,
        "argv": _fp8_argv() + ["--api-key", "secret-token"],
        "package_versions": {"torch": "2.11.0+cu130", "vllm": "0.21.0"},
    }

    stale = observed_config_from_active_manifest(
        manifest,
        active_pid=456,
        process_argv=manifest["argv"],
    )
    active = observed_config_from_active_manifest(
        manifest,
        active_pid=123,
        process_argv=manifest["argv"],
    )

    assert stale == {}
    assert active["cpu_offload_gb"] == 0.0
    assert active["torch_version"] == "2.11.0+cu130"
    assert active["runtime_argv"][-1] == "REDACTED"
    assert "secret-token" not in " ".join(active["runtime_argv"])


def test_active_manifest_rejects_different_private_path_with_fingerprint():
    manifest_argv = _fp8_argv()
    process_argv = [
        value.replace("google/gemma-4-31B-it", "/" + "home" + "/private-user/other-target")
        for value in manifest_argv
    ]
    manifest = {
        "pid": 123,
        "argv": redact_argv(manifest_argv),
        "argv_fingerprint": "not-the-process-fingerprint",
    }

    assert observed_config_from_active_manifest(
        manifest,
        active_pid=123,
        process_argv=process_argv,
    ) == {}


def test_runtime_evidence_requires_backend_port_match():
    manifest = {"pid": 123, "argv": _fp8_argv()}

    observed = observed_config_from_runtime_evidence(
        runtime_manifest=manifest,
        active_backend_pid=123,
        active_backend_argv=_fp8_argv(),
        vllm_base_url="http://vllm.local:8000",
    )

    assert observed == {}


def test_runtime_evidence_rejects_non_local_base_url_even_when_port_matches():
    observed = observed_config_from_runtime_evidence(
        runtime_manifest={"pid": 123, "argv": _fp8_argv()},
        active_backend_pid=123,
        active_backend_argv=_fp8_argv(),
        vllm_base_url="http://192.0.2.10:8000",
    )

    assert observed == {}


def test_language_model_only_mismatch_is_not_verified():
    desired = desired_config(
        replace(_fp8_profile(), language_model_only=True),
        served_model_name="gemma-4-31b-mtp",
    )
    observed = observed_config_from_process_argv(_fp8_argv())

    verification = build_config_verification(desired, observed)

    assert verification["fields"]["language_model_only"]["status"] == "mismatch"
    assert verification["status"] == "mismatch"


def test_redact_argv_handles_inline_secret_values():
    token = "hf_" + "abcdefghijklmnopqrstuvwxyz1234"
    assert redact_argv(["vllm", f"--hf-token={token}"]) == [
        "vllm",
        "--hf-token=REDACTED",
    ]


def test_redact_argv_hides_private_paths():
    assert redact_argv([
        "/" + "home" + "/private-user/env/bin/vllm",
        "--model-cache=" + "/" + "Users" + "/private-user/cache",
    ]) == [
        "REDACTED_PATH",
        "--model-cache=REDACTED_PATH",
    ]


def test_redact_argv_hides_private_paths_inside_json_values():
    path = "/" + "home" + "/private-user/private-drafter"
    assert redact_argv(["--speculative-config", f'{{"model":"{path}"}}']) == [
        "--speculative-config",
        '{"model":"REDACTED_PATH"}',
    ]


def test_redact_argv_hides_private_paths_with_spaces_inside_json_values():
    path = "/" + "home" + "/private-user/Model With Spaces/drafter"
    assert redact_argv(["--speculative-config", f'{{"model":"{path}"}}']) == [
        "--speculative-config",
        '{"model":"REDACTED_PATH"}',
    ]


def test_process_argv_redacts_private_model_paths_in_observed_fields():
    target_path = "/" + "home" + "/private-user/target"
    drafter_path = "/" + "Users" + "/private-user/drafter"
    observed = observed_config_from_process_argv([
        "vllm",
        "serve",
        target_path,
        "--speculative-config",
        f'{{"method":"mtp","model":"{drafter_path}","num_speculative_tokens":4}}',
    ])

    assert observed["target_model"] == "REDACTED_PATH"
    assert observed["drafter_model"] == "REDACTED_PATH"
    assert target_path not in " ".join(observed["runtime_argv"])
    assert drafter_path not in " ".join(observed["runtime_argv"])


def test_metrics_runtime_observation_includes_cuda_graph_evidence():
    observed = observed_config_from_metrics(
        """
vllm:spec_decode_num_draft_tokens_total 8
vllm_cuda_graph_dispatch_total 5
vllm_cuda_graph_capture_total 1
vllm_cuda_graph_capture_duration_seconds_sum 0.75
""",
        model_name="gemma-4-31b-mtp",
    )

    assert observed["cuda_graph"] == {
        "graph_metrics_registered": True,
        "graph_capture_observed": True,
        "graph_dispatch_observed": True,
        "eager_fallback_observed": False,
        "graph_dispatch_count": 5.0,
        "graph_capture_duration_seconds": 0.75,
        "graph_capture_sizes": [],
        "graph_evidence_status": "observed",
        "graph_active": True,
        "evidence_sources": ["metrics"],
    }
    assert observed["cuda_graph_observed"] is True


def test_runtime_observation_uses_cuda_graph_log_text_without_raw_log_output():
    observed = observed_config_from_metrics(
        "",
        model_name="gemma-4-31b-mtp",
        log_text="INFO Graph capturing finished in 2.5 secs, took 1.0 GiB",
    )

    assert observed["cuda_graph"]["graph_capture_observed"] is True
    assert observed["cuda_graph"]["graph_capture_duration_seconds"] == 2.5
    assert observed["cuda_graph"]["evidence_sources"] == ["logs"]
    assert observed["cuda_graph_observed"] is True
    assert observed["_sources"]["cuda_graph"] == "vllm_logs"
    assert "Graph capturing finished" not in str(observed)


def test_read_text_tail_limits_large_vllm_logs(tmp_path):
    log_path = tmp_path / "vllm.log"
    log_path.write_text(
        "ignored-prefix\n" * 20
        + "INFO Graph capturing finished in 1.0 secs\n",
        encoding="utf-8",
    )

    assert read_text_tail(log_path, max_bytes=64).endswith(
        "INFO Graph capturing finished in 1.0 secs\n"
    )


def test_cuda_graph_not_inferred_from_enforce_eager_false():
    observed = merge_observed_config(
        observed_config_from_process_argv(
            [
                "vllm",
                "serve",
                "google/gemma-4-31B-it",
                "--served-model-name",
                "gemma-4-31b-mtp",
            ]
        ),
        observed_config_from_metrics("", model_name="gemma-4-31b-mtp"),
    )

    assert observed["enforce_eager"] is False
    assert observed["cuda_graph"]["graph_evidence_status"] == "unavailable"
    assert observed["cuda_graph"]["graph_active"] is None
    assert observed["cuda_graph_observed"] is None
