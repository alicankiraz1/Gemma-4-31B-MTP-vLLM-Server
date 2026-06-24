from __future__ import annotations

from gemma4_mtp_vllm.graph_observation import parse_cuda_graph_observation


def test_cuda_graph_observation_detects_metrics_and_logs():
    metrics = """
# HELP vllm_cuda_graph_dispatch_total CUDA graph dispatches
vllm_cuda_graph_dispatch_total{model_name="gemma"} 42
vllm_cuda_graph_capture_duration_seconds_sum 1.25
vllm_cuda_graph_capture_total 3
"""
    logs = """
INFO model_runner capturing CUDA graph with sizes: [1, 2, 4]
INFO cuda graph replay/dispatch enabled
"""

    observation = parse_cuda_graph_observation(metrics_text=metrics, log_text=logs)

    assert observation == {
        "graph_metrics_registered": True,
        "graph_capture_observed": True,
        "graph_dispatch_observed": True,
        "eager_fallback_observed": False,
        "graph_dispatch_count": 42.0,
        "graph_capture_duration_seconds": 1.25,
        "graph_capture_sizes": [1, 2, 4],
        "graph_evidence_status": "observed",
        "graph_active": True,
        "evidence_sources": ["metrics", "logs"],
    }


def test_cuda_graph_observation_reports_unavailable_without_fabricating_active():
    observation = parse_cuda_graph_observation(metrics_text="", log_text="")

    assert observation["graph_metrics_registered"] is False
    assert observation["graph_capture_observed"] is False
    assert observation["graph_dispatch_observed"] is False
    assert observation["eager_fallback_observed"] is False
    assert observation["graph_dispatch_count"] is None
    assert observation["graph_capture_duration_seconds"] is None
    assert observation["graph_evidence_status"] == "unavailable"
    assert observation["graph_active"] is None


def test_cuda_graph_observation_detects_fallback_without_active():
    logs = "WARNING cudagraph fallback to eager: unsupported dynamic shape"

    observation = parse_cuda_graph_observation(metrics_text="", log_text=logs)

    assert observation["eager_fallback_observed"] is True
    assert observation["graph_evidence_status"] == "fallback_observed"
    assert observation["graph_active"] is False


def test_cuda_graph_observation_detects_metric_fallback_and_registered_idle():
    fallback = parse_cuda_graph_observation(
        metrics_text="vllm_cuda_graph_fallback_total 2\n",
    )
    idle = parse_cuda_graph_observation(
        metrics_text="# HELP vllm_cuda_graph_dispatch_total dispatches\n",
    )

    assert fallback["eager_fallback_observed"] is True
    assert fallback["graph_evidence_status"] == "fallback_observed"
    assert fallback["graph_active"] is False
    assert idle["graph_metrics_registered"] is True
    assert idle["graph_evidence_status"] == "registered_but_idle"
    assert idle["graph_active"] is None


def test_cuda_graph_observation_ignores_nonfinite_metric_values():
    observation = parse_cuda_graph_observation(
        metrics_text=(
            "vllm_cuda_graph_dispatch_total NaN\n"
            "vllm_cuda_graph_capture_duration_seconds_sum inf\n"
        ),
    )

    assert observation["graph_metrics_registered"] is False
    assert observation["graph_dispatch_observed"] is False
    assert observation["graph_capture_observed"] is False
    assert observation["graph_evidence_status"] == "unavailable"
    assert observation["graph_active"] is None
