from __future__ import annotations

import math
import statistics
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkEndpointResult:
    e2e_output_tokens_per_second: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    inter_token_latency_ms_p50: float | None
    inter_token_latency_ms_p95: float | None
    total_latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_output_token_ids: list[int] | None = None
    reasoning_text: str = ""
    visible_content: str = ""
    token_timing_basis: str = "unavailable"
    tpot_basis: str = "unavailable"
    itl_basis: str = "unavailable"
    stream_chunk_interval_ms_p50: float | None = None
    stream_chunk_interval_ms_p95: float | None = None
    timing_evidence_valid: bool = False
    token_count_validation_status: str = "raw_token_ids_missing"
    raw_stream_chunks: list[dict[str, Any]] | None = None
    transport_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkObservation:
    index: int
    baseline: BenchmarkEndpointResult
    mtp: BenchmarkEndpointResult
    speedup: float | None
    deterministic_parity: bool | None
    parity_basis: str
    parity_failure: bool
    mtp_metrics_before: dict[str, Any] | None = None
    mtp_metrics_after: dict[str, Any] | None = None
    mtp_metrics_delta: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    profile: str
    prompt_name: str
    prompt: str
    output_token_target: int
    num_speculative_tokens: int
    observations: list[BenchmarkObservation]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["observations"] = [asdict(obs) for obs in self.observations]
        data["statistics"] = benchmark_statistics(self.observations)
        failure_reasons = benchmark_failure_reasons(self.observations)
        data["failed"] = bool(failure_reasons)
        data["failure_reasons"] = failure_reasons
        return data


def speedup(no_draft_tps: float | None, mtp_tps: float | None) -> float | None:
    if no_draft_tps is None or mtp_tps is None:
        return None
    if not math.isfinite(no_draft_tps) or not math.isfinite(mtp_tps):
        return None
    if no_draft_tps <= 0:
        return None
    ratio = mtp_tps / no_draft_tps
    return ratio if math.isfinite(ratio) else None


def deterministic_parity(
    no_draft_text: str,
    mtp_text: str,
    *,
    temperature: float,
    top_p: float,
) -> bool | None:
    if temperature != 0.0 or top_p != 1.0:
        return None
    return no_draft_text == mtp_text


def median_optional(values: list[float | None]) -> float | None:
    cleaned = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not cleaned:
        return None
    return statistics.median(cleaned)


def benchmark_statistics(observations: list[BenchmarkObservation]) -> dict[str, Any]:
    return {
        "baseline": {
            "e2e_output_tokens_per_second": metric_summary(
                [
                    obs.baseline.e2e_output_tokens_per_second
                    for obs in observations
                ]
            ),
        },
        "mtp": {
            "e2e_output_tokens_per_second": metric_summary(
                [obs.mtp.e2e_output_tokens_per_second for obs in observations]
            ),
        },
        "speedup": metric_summary([obs.speedup for obs in observations]),
    }


def benchmark_failure_reasons(
    observations: list[BenchmarkObservation],
) -> list[str]:
    reasons: list[str] = []
    if any(obs.parity_failure for obs in observations):
        reasons.append("deterministic_parity_failed")
    return reasons


def metric_summary(
    values: list[float | None],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 1,
) -> dict[str, Any]:
    cleaned = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not cleaned:
        return {
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "bootstrap_ci_95": {"low": None, "high": None},
        }
    ci = bootstrap_ci(cleaned, samples=bootstrap_samples, seed=seed)
    return {
        "median": statistics.median(cleaned),
        "p10": percentile(cleaned, 10),
        "p90": percentile(cleaned, 90),
        "p95": percentile(cleaned, 95),
        "bootstrap_ci_95": (
            {"low": ci[0], "high": ci[1]}
            if ci is not None
            else {"low": None, "high": None}
        ),
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float | None],
    *,
    samples: int = 1000,
    seed: int = 1,
) -> tuple[float, float] | None:
    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return (cleaned[0], cleaned[0])
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        sample = [rng.choice(cleaned) for _ in cleaned]
        medians.append(statistics.median(sample))
    low = percentile(medians, 2.5)
    high = percentile(medians, 97.5)
    if low is None or high is None:
        return None
    return (low, high)
