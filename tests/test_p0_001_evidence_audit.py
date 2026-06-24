from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/p0_001_evidence_audit.py")
    spec = importlib.util.spec_from_file_location("p0_001_evidence_audit", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bench_payload(
    *,
    label: str,
    profile: str,
    return_token_ids: bool | None,
    tokens: list[int],
    visible_content: str | None = None,
    raw_tokens: bool = False,
) -> dict:
    request_body = {
        "model": "gemma-4-31b-mtp",
        "max_tokens": len(tokens),
        "stream": True,
    }
    if return_token_ids is not None:
        request_body["return_token_ids"] = return_token_ids
    observation = {
        "index": 1,
        "output_sha256": "a" * 64,
        "output_token_ids": tokens,
        "parity_ready": True,
        "tokenization_status": "available",
        "result": {},
    }
    if visible_content is not None:
        observation["visible_content"] = visible_content
        observation["result"]["visible_content"] = visible_content
    if raw_tokens:
        observation["raw_output_token_ids"] = tokens
        observation["timing_evidence_valid"] = True
        observation["tokenization_status"] = "matched"
        observation["result"]["raw_output_token_ids"] = tokens
        observation["result"]["timing_evidence_valid"] = True
    return {
        "benchmark_protocol_version": 2,
        "benchmark_kind": "single_endpoint_runtime",
        "status": "complete",
        "label": label,
        "profile": profile,
        "groups": [
            {
                "prompt_name": "prompt_1",
                "prompt": "summarize",
                "output_token_target": len(tokens),
                "request_body": request_body,
                "observations": [observation, dict(observation)],
            }
        ],
    }


def _write_payload(path: Path, audit, payload: dict) -> None:
    path.write_text(audit.json.dumps(payload), encoding="utf-8")


def test_legacy_b_vs_d_is_labeled_retokenized_visible_output(tmp_path):
    audit = _load_module()
    control = tmp_path / "eager-true.json"
    candidate = tmp_path / "eager-false.json"
    audit_payload_control = _bench_payload(
        label="eager_true",
        profile="tp2_2x32_fp8_gpuonly",
        return_token_ids=None,
        tokens=[1, 2, 3, 4],
    )
    audit_payload_candidate = _bench_payload(
        label="eager_false_cuda_graph",
        profile="tp2_2x32_fp8_gpuonly_cuda_graph",
        return_token_ids=None,
        tokens=[1, 2, 9, 4],
    )
    _write_payload(control, audit, audit_payload_control)
    _write_payload(candidate, audit, audit_payload_candidate)

    result = audit.analyze_b_vs_d_pair(
        label="legacy",
        control_path=control,
        candidate_path=candidate,
    )

    target = result["target_diagnostics"][0]
    assert result["raw_generation_parity_claim"] is False
    assert target["sequence_source"] == "retokenized_visible_output"
    assert (
        target["parity_label"]
        == "retokenized_visible_output_cross_mode_diagnostic"
    )
    assert target["within_control_repeatability"]["status"] == "passed"
    assert target["within_candidate_repeatability"]["status"] == "passed"
    assert target["cross_mode_token_diagnostic"] == {
        "exact_match": False,
        "control_length": 4,
        "candidate_length": 4,
        "output_length_equal": True,
        "longest_common_prefix_tokens": 2,
        "first_divergence_position_0_based": 2,
        "control_token_at_first_divergence": 3,
        "candidate_token_at_first_divergence": 9,
        "same_position_token_matches": 3,
        "matching_token_percentage_same_position_over_max_len": 0.75,
    }
    assert target["normalized_visible_answer_similarity"]["status"] == "unavailable"


def test_raw_stream_token_ids_and_visible_similarity_are_reported(tmp_path):
    audit = _load_module()
    control = tmp_path / "eager_mtp.json"
    candidate = tmp_path / "graph_mtp.json"
    _write_payload(
        control,
        audit,
        _bench_payload(
            label="eager_mtp",
            profile="tp2_2x32_fp8_gpuonly",
            return_token_ids=True,
            tokens=[5, 6, 7],
            visible_content="Final answer: Local Gemma trades speed for control.",
            raw_tokens=True,
        ),
    )
    _write_payload(
        candidate,
        audit,
        _bench_payload(
            label="graph_mtp",
            profile="tp2_2x32_fp8_gpuonly_cuda_graph",
            return_token_ids=True,
            tokens=[5, 6, 8],
            visible_content="Final answer: Local Gemma trades speed for control!",
            raw_tokens=True,
        ),
    )

    result = audit.analyze_b_vs_d_pair(
        label="repair",
        control_path=control,
        candidate_path=candidate,
    )

    target = result["target_diagnostics"][0]
    assert target["sequence_source"] == "raw_stream_token_ids"
    assert target["parity_label"] == "raw_token_ids_cross_mode_diagnostic"
    assert target["normalized_visible_answer_similarity"]["status"] == "computed"
    assert 0.95 < target["normalized_visible_answer_similarity"]["ratio"] < 1.0


def test_missing_file_and_inaccessible_roots_are_invalid(tmp_path, monkeypatch):
    audit = _load_module()
    missing = audit.build_evidence_root_index(
        label="missing",
        root=tmp_path / "missing",
        include_file_index=True,
    )
    assert missing["status"] == "invalid"
    assert missing["file_count"] == 0
    assert missing["errors"] == [{"relative_path": ".", "reason": "root_missing"}]

    file_root = tmp_path / "not-a-dir"
    file_root.write_text("not a directory\n", encoding="utf-8")
    file_result = audit.build_evidence_root_index(
        label="file",
        root=file_root,
        include_file_index=True,
    )
    assert file_result["status"] == "invalid"
    assert file_result["errors"] == [
        {"relative_path": ".", "reason": "root_not_directory"}
    ]

    directory = tmp_path / "no-access"
    directory.mkdir()
    monkeypatch.setattr(audit.os, "access", lambda _path, _mode: False)
    inaccessible = audit.build_evidence_root_index(
        label="no-access",
        root=directory,
        include_file_index=True,
    )
    assert inaccessible["status"] == "inaccessible"
    assert inaccessible["errors"] == [
        {"relative_path": ".", "reason": "root_inaccessible"}
    ]


def test_disappearing_file_is_recorded_as_error(tmp_path, monkeypatch):
    audit = _load_module()
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "gone.txt").write_text("content\n", encoding="utf-8")

    def raise_disappeared(_path):
        raise OSError("disappeared")

    monkeypatch.setattr(audit, "_sha256_file", raise_disappeared)

    result = audit.build_evidence_root_index(
        label="sample",
        root=root,
        include_file_index=True,
    )

    assert result["status"] == "complete_with_errors"
    assert result["file_count"] == 0
    assert result["skipped"] == [
        {"relative_path": "gone.txt", "reason": "sha256_read_error"}
    ]
    assert result["errors"] == [
        {
            "relative_path": "gone.txt",
            "reason": "sha256_read_error",
            "error_type": "OSError",
        }
    ]


def test_partial_raw_and_retokenized_group_is_mixed_source(tmp_path):
    audit = _load_module()
    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    payload = _bench_payload(
        label="partial",
        profile="tp2_2x32_fp8_gpuonly",
        return_token_ids=True,
        tokens=[1, 2, 3],
        raw_tokens=True,
    )
    payload["groups"][0]["observations"][1] = {
        "index": 2,
        "output_sha256": "b" * 64,
        "output_token_ids": [1, 2, 3],
        "parity_ready": True,
        "tokenization_status": "available",
        "result": {},
    }
    _write_payload(control, audit, payload)
    _write_payload(candidate, audit, payload)

    result = audit.analyze_b_vs_d_pair(
        label="partial",
        control_path=control,
        candidate_path=candidate,
    )

    target = result["target_diagnostics"][0]
    assert target["sequence_source"] == "mixed_token_sources"
    assert target["parity_label"] == "mixed_token_sources_cross_mode_diagnostic"
    assert target["within_control_repeatability"]["status"] == "mixed_token_sources"
    assert target["cross_mode_token_diagnostic"]["status"] == "mixed_token_sources"
    assert target["raw_generation_parity_claim"] is False


def test_raw_vs_retokenized_pair_is_mixed_source_not_raw_parity(tmp_path):
    audit = _load_module()
    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    _write_payload(
        control,
        audit,
        _bench_payload(
            label="raw",
            profile="tp2_2x32_fp8_gpuonly",
            return_token_ids=True,
            tokens=[1, 2, 3],
            raw_tokens=True,
        ),
    )
    _write_payload(
        candidate,
        audit,
        _bench_payload(
            label="retokenized",
            profile="tp2_2x32_fp8_gpuonly_cuda_graph",
            return_token_ids=None,
            tokens=[1, 2, 3],
        ),
    )

    result = audit.analyze_b_vs_d_pair(
        label="mixed",
        control_path=control,
        candidate_path=candidate,
    )

    target = result["target_diagnostics"][0]
    assert target["control_sequence_source"] == "raw_stream_token_ids"
    assert target["candidate_sequence_source"] == "retokenized_visible_output"
    assert target["sequence_source"] == "mixed_token_sources"
    assert target["parity_label"] == "mixed_token_sources_cross_mode_diagnostic"
    assert target["cross_mode_token_diagnostic"] == {
        "status": "mixed_token_sources",
        "reason": "token sources are not comparable",
    }


def test_generated_at_override_is_deterministic():
    audit = _load_module()

    payload = audit.build_audit_payload(
        evidence_roots=[],
        pairs=[],
        generated_at_utc="2026-06-24T00:00:00+00:00",
    )

    assert payload["generated_at_utc"] == "2026-06-24T00:00:00+00:00"


def test_local_path_scan_matches_volumes_spaces_and_general_unix_paths(tmp_path):
    audit = _load_module()
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "paths.txt").write_text(
        "\n".join(
            [
                "/" + "Volumes/My Drive/Gemma run/output.json",
                "/" + "srv/model-cache/run.json",
                "/" + "Users/Alice/Dir With Space/file.txt",
            ]
        ),
        encoding="utf-8",
    )

    result = audit.build_evidence_root_index(
        label="paths",
        root=root,
        include_file_index=True,
    )

    assert result["local_path_scan"]["match_count"] == 3
    assert result["local_path_scan"]["matched_files"] == ["paths.txt"]


def test_evidence_index_records_hashes_and_sanitized_scan_counts(tmp_path):
    audit = _load_module()
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "clean.txt").write_text("hello\n", encoding="utf-8")
    (root / "scan.txt").write_text(
        "/" + "home/homelander/private " + "hf_" + "abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )

    result = audit.build_evidence_root_index(
        label="sample",
        root=root,
        include_file_index=True,
    )

    assert result["file_count"] == 2
    assert result["index_sha256"]
    assert result["files"][0]["sha256"]
    assert result["secret_scan"] == {
        "match_count": 1,
        "matched_file_count": 1,
        "matched_files": ["scan.txt"],
        "scanned_file_count": 2,
        "scan_error_count": 0,
        "scan_error_files": [],
        "scan_status": "complete",
        "scan_mode": "streaming",
        "snippets_included": False,
    }
    assert result["local_path_scan"] == {
        "match_count": 1,
        "matched_file_count": 1,
        "matched_files": ["scan.txt"],
        "scanned_file_count": 2,
        "scan_error_count": 0,
        "scan_error_files": [],
        "scan_status": "complete",
        "scan_mode": "streaming",
        "snippets_included": False,
    }
