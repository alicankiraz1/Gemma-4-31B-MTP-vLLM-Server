from __future__ import annotations

from pathlib import Path

import pytest

from gemma4_mtp_vllm.mtp_metrics import mtp_metric_delta, parse_mtp_metrics


def test_parse_real_vllm_0_21_mtp_metrics_fixture():
    text = Path("tests/fixtures/vllm_0_21_mtp_metrics.prom").read_text()

    mtp = parse_mtp_metrics(text)

    assert mtp["state"] == "active"
    assert mtp["metrics_registered"] is True
    assert mtp["active_since_start"] is True
    assert mtp["draft_rounds_total"] == 240.0
    assert mtp["drafted_tokens_total"] == 960.0
    assert mtp["accepted_tokens_total"] == 863.0
    assert mtp["rejected_tokens_total"] == 97.0
    assert mtp["acceptance_rate"] == pytest.approx(863.0 / 960.0)
    assert mtp["mean_acceptance_length"] == pytest.approx(863.0 / 240.0)


def test_registered_zero_metrics_are_idle_not_active():
    mtp = parse_mtp_metrics(
        'vllm:spec_decode_num_drafts_total{model_name="m"} 0\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="m"} 0\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 0\n'
    )

    assert mtp["state"] == "registered_but_idle"
    assert mtp["metrics_registered"] is True
    assert mtp["active_since_start"] is False


def test_metric_names_without_values_are_unavailable_not_active():
    mtp = parse_mtp_metrics("# HELP vllm:spec_decode_num_draft_tokens_total present\n")

    assert mtp["state"] == "unavailable"
    assert mtp["metrics_registered"] is False


def test_parser_filters_by_model_name_label():
    mtp = parse_mtp_metrics(
        'vllm:spec_decode_num_drafts_total{model_name="other"} 2\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="other"} 8\n'
        'vllm:spec_decode_num_drafts_total{model_name="target"} 0\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="target"} 0\n',
        model_name="target",
    )

    assert mtp["state"] == "registered_but_idle"
    assert mtp["drafted_tokens_total"] == 0.0


def test_parse_error_is_observable_for_relevant_metric_value():
    mtp = parse_mtp_metrics("vllm:spec_decode_num_draft_tokens_total not-a-number\n")

    assert mtp["state"] == "parse_error"
    assert "not-a-number" in mtp["parse_error"]


def test_positive_drafted_token_delta_is_active():
    before = parse_mtp_metrics(
        "vllm:spec_decode_num_drafts_total 10\n"
        "vllm:spec_decode_num_draft_tokens_total 40\n"
        "vllm:spec_decode_num_accepted_tokens_total 20\n"
    )
    after = parse_mtp_metrics(
        "vllm:spec_decode_num_drafts_total 12\n"
        "vllm:spec_decode_num_draft_tokens_total 48\n"
        "vllm:spec_decode_num_accepted_tokens_total 25\n"
    )

    delta = mtp_metric_delta(before, after)

    assert delta["state"] == "active"
    assert delta["draft_rounds_delta"] == 2.0
    assert delta["drafted_tokens_delta"] == 8.0
    assert delta["accepted_tokens_delta"] == 5.0
