from __future__ import annotations

import json

import pytest

from gemma4_mtp_vllm.benchmarking import (
    BenchmarkEndpointResult,
    BenchmarkObservation,
    BenchmarkSummary,
    bootstrap_ci,
    deterministic_parity,
    median_optional,
    metric_summary,
    speedup,
)


def test_speedup_returns_ratio_when_both_positive():
    assert speedup(10.0, 20.0) == pytest.approx(2.0)


def test_speedup_returns_none_when_baseline_missing():
    assert speedup(0.0, 10.0) is None
    assert speedup(None, 10.0) is None
    assert speedup(10.0, None) is None
    assert speedup(float("nan"), 10.0) is None
    assert speedup(10.0, float("inf")) is None
    assert speedup(1e-308, 1e308) is None


@pytest.mark.parametrize(
    "no_draft, mtp, expected",
    [
        ("hello", "hello", True),
        ("hello", "world", False),
    ],
)
def test_deterministic_parity_greedy(no_draft, mtp, expected):
    assert deterministic_parity(no_draft, mtp, temperature=0.0, top_p=1.0) is expected


def test_deterministic_parity_returns_none_when_sampling():
    assert deterministic_parity("a", "b", temperature=0.7, top_p=1.0) is None


def test_median_optional_returns_value():
    assert median_optional([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert median_optional([]) is None
    assert median_optional([None]) is None
    assert median_optional([float("nan"), float("inf")]) is None


def test_benchmark_summary_to_json_roundtrip():
    baseline = BenchmarkEndpointResult(
        e2e_output_tokens_per_second=10.0,
        ttft_ms=12.0,
        tpot_ms=5.0,
        inter_token_latency_ms_p50=4.0,
        inter_token_latency_ms_p95=6.0,
        total_latency_ms=100.0,
        prompt_tokens=3,
        completion_tokens=8,
    )
    mtp = BenchmarkEndpointResult(
        e2e_output_tokens_per_second=20.0,
        ttft_ms=10.0,
        tpot_ms=3.0,
        inter_token_latency_ms_p50=2.0,
        inter_token_latency_ms_p95=4.0,
        total_latency_ms=80.0,
        prompt_tokens=3,
        completion_tokens=8,
    )
    summary = BenchmarkSummary(
        profile="safe80",
        prompt_name="default",
        prompt="Hello",
        output_token_target=8,
        num_speculative_tokens=4,
        observations=[
            BenchmarkObservation(
                index=1,
                baseline=baseline,
                mtp=mtp,
                speedup=2.0,
                deterministic_parity=True,
                parity_basis="token",
                parity_failure=False,
            )
        ],
    )
    body = json.loads(json.dumps(summary.to_dict()))
    serialized = json.dumps(body)
    assert body["profile"] == "safe80"
    assert body["output_token_target"] == 8
    assert body["observations"][0]["speedup"] == 2.0
    assert body["observations"][0]["baseline"]["e2e_output_tokens_per_second"] == 10.0
    assert body["statistics"]["mtp"]["e2e_output_tokens_per_second"]["median"] == 20.0
    assert "generation_tps" not in serialized


def test_metric_summary_reports_percentiles_and_bootstrap_ci():
    summary = metric_summary([10.0, 20.0, 30.0], bootstrap_samples=200, seed=7)

    assert summary["median"] == pytest.approx(20.0)
    assert summary["p10"] == pytest.approx(12.0)
    assert summary["p90"] == pytest.approx(28.0)
    assert summary["p95"] == pytest.approx(29.0)
    assert summary["bootstrap_ci_95"]["low"] <= summary["median"]
    assert summary["bootstrap_ci_95"]["high"] >= summary["median"]


def test_metric_summary_ignores_nonfinite_values():
    summary = metric_summary([10.0, float("nan"), float("inf")])

    assert summary["median"] == 10.0


def test_bootstrap_ci_returns_none_for_empty_values():
    assert bootstrap_ci([], samples=100, seed=1) is None


def test_benchmark_summary_records_parity_failure():
    endpoint = BenchmarkEndpointResult(
        e2e_output_tokens_per_second=10.0,
        ttft_ms=1.0,
        tpot_ms=1.0,
        inter_token_latency_ms_p50=1.0,
        inter_token_latency_ms_p95=1.0,
        total_latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=2,
    )
    summary = BenchmarkSummary(
        profile="safe80",
        prompt_name="default",
        prompt="Hello",
        output_token_target=2,
        num_speculative_tokens=4,
        observations=[
            BenchmarkObservation(
                index=1,
                baseline=endpoint,
                mtp=endpoint,
                speedup=1.0,
                deterministic_parity=False,
                parity_basis="token",
                parity_failure=True,
            )
        ],
    )

    body = summary.to_dict()

    assert body["failed"] is True
    assert body["failure_reasons"] == ["deterministic_parity_failed"]
