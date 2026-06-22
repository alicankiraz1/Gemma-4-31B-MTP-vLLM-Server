from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkObservation:
    index: int
    no_draft_generation_tps: float | None
    mtp_generation_tps: float | None
    speedup: float | None
    deterministic_parity: bool | None
    mtp_metrics_before: dict[str, Any] | None = None
    mtp_metrics_after: dict[str, Any] | None = None
    mtp_metrics_delta: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    profile: str
    prompt_name: str
    prompt: str
    num_speculative_tokens: int
    observations: list[BenchmarkObservation]
    median_no_draft_generation_tps: float | None
    median_mtp_generation_tps: float | None
    median_speedup: float | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["observations"] = [asdict(obs) for obs in self.observations]
        return data


def speedup(no_draft_tps: float | None, mtp_tps: float | None) -> float | None:
    if no_draft_tps is None or mtp_tps is None:
        return None
    if no_draft_tps <= 0:
        return None
    return mtp_tps / no_draft_tps


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
    cleaned = [v for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return statistics.median(cleaned)
