from __future__ import annotations

import re
from typing import Any


GRAPH_METRIC_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*graph[A-Za-z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$",
    re.I,
)
GRAPH_CAPTURE_SIZE_RE = re.compile(
    r"(?i)(?:cuda\s*graph|cudagraph).{0,80}?(?:size|sizes|batch sizes?)"
    r"[^0-9]+(?P<sizes>[0-9][0-9,\s\[\]]*)"
)
GRAPH_CAPTURE_RE = re.compile(r"(?i)(?:cuda\s*graph|cudagraph).{0,80}captur")
GRAPH_DISPATCH_RE = re.compile(
    r"(?i)(?:cuda\s*graph|cudagraph).{0,80}(?:dispatch|replay)"
)
GRAPH_FALLBACK_RE = re.compile(
    r"(?i)(?:cuda\s*graph|cudagraph).{0,120}(?:fallback|falling back|miss)"
)


def parse_cuda_graph_observation(
    *,
    metrics_text: str,
    log_text: str = "",
) -> dict[str, Any]:
    metric_summary = _parse_graph_metrics(metrics_text)
    log_summary = _parse_graph_logs(log_text)

    graph_metrics_registered = metric_summary["registered"]
    graph_capture_observed = bool(
        metric_summary["capture_observed"] or log_summary["capture_observed"]
    )
    graph_dispatch_observed = bool(
        metric_summary["dispatch_observed"] or log_summary["dispatch_observed"]
    )
    eager_fallback_observed = bool(
        metric_summary["fallback_observed"] or log_summary["fallback_observed"]
    )

    sources = []
    if graph_metrics_registered:
        sources.append("metrics")
    if log_summary["has_graph_log"]:
        sources.append("logs")

    if eager_fallback_observed:
        evidence_status = "fallback_observed"
        graph_active: bool | None = False
    elif graph_capture_observed or graph_dispatch_observed:
        evidence_status = "observed"
        graph_active = True
    elif graph_metrics_registered or log_summary["has_graph_log"]:
        evidence_status = "registered_but_idle"
        graph_active = None
    else:
        evidence_status = "unavailable"
        graph_active = None

    return {
        "graph_metrics_registered": graph_metrics_registered,
        "graph_capture_observed": graph_capture_observed,
        "graph_dispatch_observed": graph_dispatch_observed,
        "eager_fallback_observed": eager_fallback_observed,
        "graph_dispatch_count": metric_summary["dispatch_count"],
        "graph_capture_duration_seconds": metric_summary["capture_duration_seconds"],
        "graph_capture_sizes": log_summary["capture_sizes"],
        "graph_evidence_status": evidence_status,
        "graph_active": graph_active,
        "evidence_sources": sources,
    }


def _parse_graph_metrics(metrics_text: str) -> dict[str, Any]:
    registered = False
    capture_observed = False
    dispatch_observed = False
    fallback_observed = False
    dispatch_count: float | None = None
    capture_duration_seconds: float | None = None

    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if "graph" in stripped.lower():
                registered = True
            continue
        match = GRAPH_METRIC_RE.match(stripped)
        if not match:
            continue
        name = match.group("name").lower().replace(":", "_")
        value = _float_or_none(match.group("value"))
        if value is None:
            continue
        registered = True
        if "dispatch" in name or "replay" in name:
            dispatch_count = _sum_optional(dispatch_count, value)
            if value > 0:
                dispatch_observed = True
        if "capture" in name:
            if "duration" in name or "seconds" in name:
                capture_duration_seconds = _sum_optional(
                    capture_duration_seconds,
                    value,
                )
            if value > 0:
                capture_observed = True
        if "fallback" in name or "miss" in name:
            if value > 0:
                fallback_observed = True

    return {
        "registered": registered,
        "capture_observed": capture_observed,
        "dispatch_observed": dispatch_observed,
        "fallback_observed": fallback_observed,
        "dispatch_count": dispatch_count,
        "capture_duration_seconds": capture_duration_seconds,
    }


def _parse_graph_logs(log_text: str) -> dict[str, Any]:
    capture_observed = bool(GRAPH_CAPTURE_RE.search(log_text))
    dispatch_observed = bool(GRAPH_DISPATCH_RE.search(log_text))
    fallback_observed = bool(GRAPH_FALLBACK_RE.search(log_text))
    capture_sizes = _capture_sizes_from_logs(log_text)
    return {
        "has_graph_log": bool(
            capture_observed
            or dispatch_observed
            or fallback_observed
            or capture_sizes
        ),
        "capture_observed": capture_observed,
        "dispatch_observed": dispatch_observed,
        "fallback_observed": fallback_observed,
        "capture_sizes": capture_sizes,
    }


def _capture_sizes_from_logs(log_text: str) -> list[int]:
    sizes: set[int] = set()
    for match in GRAPH_CAPTURE_SIZE_RE.finditer(log_text):
        for raw in re.findall(r"\d+", match.group("sizes")):
            value = int(raw)
            if value > 0:
                sizes.add(value)
    return sorted(sizes)


def _float_or_none(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _sum_optional(left: float | None, right: float) -> float:
    return (left or 0.0) + right
