from __future__ import annotations

from datetime import UTC, datetime
import math
import re
from typing import Any


METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)


def parse_mtp_metrics(
    metrics_text: str,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    totals = {
        "draft_rounds_total": 0.0,
        "drafted_tokens_total": 0.0,
        "accepted_tokens_total": 0.0,
        "speculative_decode_request_count": 0.0,
    }
    registered = False
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        kind = _metric_kind(name)
        if kind is None:
            continue
        labels = _parse_labels(match.group("labels") or "")
        if model_name is not None and labels.get("model_name") not in {None, model_name}:
            continue
        registered = True
        try:
            value = float(match.group("value"))
        except ValueError:
            return _empty_mtp_state(
                "parse_error",
                metrics_registered=True,
                parse_error=f"could not parse metric value: {match.group('value')}",
            )
        if not math.isfinite(value):
            return _empty_mtp_state(
                "parse_error",
                metrics_registered=True,
                parse_error=f"non-finite metric value: {match.group('value')}",
            )
        totals[kind] += value

    if not registered:
        return _empty_mtp_state("unavailable", metrics_registered=False)

    drafted = totals["drafted_tokens_total"]
    accepted = totals["accepted_tokens_total"]
    rounds = totals["draft_rounds_total"]
    active_since_start = drafted > 0 or rounds > 0
    state = "active" if active_since_start else "registered_but_idle"
    rejected = max(drafted - accepted, 0.0) if drafted > 0 else 0.0
    result = _empty_mtp_state(
        state,
        metrics_registered=True,
        active_since_start=active_since_start,
    )
    result.update(
        {
            **totals,
            "rejected_tokens_total": rejected,
            "acceptance_rate": accepted / drafted if drafted > 0 else None,
            "mean_acceptance_length": accepted / rounds if rounds > 0 else None,
        }
    )
    return result


def mtp_metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before.get("state") == "parse_error" or after.get("state") == "parse_error":
        return _empty_delta("parse_error")
    if not before.get("metrics_registered") or not after.get("metrics_registered"):
        return _empty_delta("unavailable")
    draft_rounds = _delta(before, after, "draft_rounds_total")
    drafted = _delta(before, after, "drafted_tokens_total")
    accepted = _delta(before, after, "accepted_tokens_total")
    rejected = max(drafted - accepted, 0.0) if drafted > 0 else 0.0
    active = drafted > 0 or draft_rounds > 0
    return {
        "state": "active" if active else "registered_but_idle",
        "draft_rounds_delta": draft_rounds,
        "drafted_tokens_delta": drafted,
        "accepted_tokens_delta": accepted,
        "rejected_tokens_delta": rejected,
        "acceptance_rate_delta": accepted / drafted if drafted > 0 else None,
        "mean_acceptance_length_delta": accepted / draft_rounds if draft_rounds > 0 else None,
    }


def _metric_kind(name: str) -> str | None:
    normalized = name.lower().replace(":", "_")
    if normalized.endswith("_created"):
        return None
    if "spec_decode" not in normalized and "speculative" not in normalized:
        return None
    if "accepted_tokens_per_pos" in normalized:
        return None
    if "draft_tokens" in normalized and normalized.endswith("_total"):
        return "drafted_tokens_total"
    if "accepted_tokens" in normalized and normalized.endswith("_total"):
        return "accepted_tokens_total"
    if "num_drafts" in normalized and normalized.endswith("_total"):
        return "draft_rounds_total"
    if "request" in normalized and normalized.endswith("_total"):
        return "speculative_decode_request_count"
    return None


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def _empty_mtp_state(
    state: str,
    *,
    metrics_registered: bool,
    active_since_start: bool = False,
    parse_error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": state,
        "metrics_registered": metrics_registered,
        "active_since_start": active_since_start,
        "last_observation": datetime.now(UTC).isoformat(),
        "draft_rounds_total": None,
        "drafted_tokens_total": None,
        "accepted_tokens_total": None,
        "rejected_tokens_total": None,
        "acceptance_rate": None,
        "mean_acceptance_length": None,
        "speculative_decode_request_count": None,
    }
    if parse_error is not None:
        result["parse_error"] = parse_error
    return result


def _empty_delta(state: str) -> dict[str, Any]:
    return {
        "state": state,
        "draft_rounds_delta": None,
        "drafted_tokens_delta": None,
        "accepted_tokens_delta": None,
        "rejected_tokens_delta": None,
        "acceptance_rate_delta": None,
        "mean_acceptance_length_delta": None,
    }


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float:
    before_value = before.get(key)
    after_value = after.get(key)
    if not isinstance(before_value, (int, float)):
        return 0.0
    if not isinstance(after_value, (int, float)):
        return 0.0
    return max(float(after_value) - float(before_value), 0.0)
