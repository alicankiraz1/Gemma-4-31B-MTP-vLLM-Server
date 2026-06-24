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
    control.write_text(audit.json.dumps(audit_payload_control), encoding="utf-8")
    candidate.write_text(audit.json.dumps(audit_payload_candidate), encoding="utf-8")

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
    control.write_text(
        audit.json.dumps(
            _bench_payload(
                label="eager_mtp",
                profile="tp2_2x32_fp8_gpuonly",
                return_token_ids=True,
                tokens=[5, 6, 7],
                visible_content="Final answer: Local Gemma trades speed for control.",
                raw_tokens=True,
            )
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        audit.json.dumps(
            _bench_payload(
                label="graph_mtp",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                return_token_ids=True,
                tokens=[5, 6, 8],
                visible_content="Final answer: Local Gemma trades speed for control!",
                raw_tokens=True,
            )
        ),
        encoding="utf-8",
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
        "snippets_included": False,
    }
    assert result["local_path_scan"] == {
        "match_count": 1,
        "matched_file_count": 1,
        "matched_files": ["scan.txt"],
        "snippets_included": False,
    }
