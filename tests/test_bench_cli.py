from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from gemma4_mtp_vllm import cli as cli_module
from gemma4_mtp_vllm.cli import app


runner = CliRunner()


class _FakeBenchClock:
    def __init__(self) -> None:
        self.now = 0.0
        self._start = 0.0
        self._scheduled: list[float] = []

    def perf_counter(self) -> float:
        if self._scheduled:
            value = self._scheduled.pop(0)
            if not self._scheduled:
                self.now = value + 1.0
            return value
        self._start = self.now
        return self.now

    def schedule(self, *, completion_tokens: int, tps: float) -> None:
        elapsed = completion_tokens / tps
        self._scheduled = [
            self._start + (elapsed * (index + 1) / completion_tokens)
            for index in range(completion_tokens)
        ]
        self._scheduled.append(self._start + elapsed)
        self._scheduled.append(self._start + elapsed)


def _install_fake_clock(monkeypatch) -> _FakeBenchClock:
    clock = _FakeBenchClock()
    monkeypatch.setattr("gemma4_mtp_vllm.cli.time.perf_counter", clock.perf_counter)
    return clock


def _stream_response(
    *,
    text: str,
    extensions: dict[str, object] | None = None,
) -> httpx.Response:
    chunks = "".join(
        (
            'data: {"choices":[{"delta":'
            f'{{"content":{json.dumps(char)},"token_ids":[{ord(char)}]}}'
            "}]}\n\n"
        )
        for char in text
    )
    body = (
        chunks +
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
        extensions=extensions or {},
    )


class _TrackingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[httpx.Request] = []
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    async def aclose(self) -> None:
        self.closed = True


class _CloseProbe:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        if self.failure is not None:
            raise self.failure


def _make_handler(
    *,
    tps_mtp: float,
    tps_baseline: float,
    clock: _FakeBenchClock | None = None,
    captured_bodies: list[dict] | None = None,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            text = str(body.get("prompt") or "")
            return httpx.Response(
                200,
                json={"count": len(text), "tokens": [ord(char) for char in text]},
            )
        # Headers carry which URL was hit; we route by host.
        host = request.url.host
        tps = tps_mtp if host and host.startswith("mtp") else tps_baseline
        if clock is not None:
            clock.schedule(completion_tokens=8, tps=tps)
        if captured_bodies is not None:
            captured_bodies.append(json.loads(request.content))
        return _stream_response(text="response")

    return handler


def test_bench_command_emits_summary(monkeypatch, tmp_path):
    captured_bodies: list[dict] = []
    clock = _install_fake_clock(monkeypatch)
    handler = _make_handler(
        tps_mtp=20.0,
        tps_baseline=10.0,
        clock=clock,
        captured_bodies=captured_bodies,
    )
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--max-tokens",
            "8",
            "--runs",
            "2",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(out.read_text())
    serialized = json.dumps(body)
    assert body["profile"] == "safe80"
    assert body["output_token_target"] == 8
    assert len(body["observations"]) == 2
    assert body["statistics"]["speedup"]["median"] == pytest.approx(2.0)
    assert body["observations"][0]["mtp"]["e2e_output_tokens_per_second"] == 20.0
    assert body["observations"][0]["mtp"]["ttft_ms"] is not None
    assert body["observations"][0]["mtp"]["tpot_ms"] is not None
    assert body["observations"][0]["parity_basis"] == "token"
    assert "generation_tps" not in serialized
    assert captured_bodies
    assert all(request_body["stream"] is True for request_body in captured_bodies)
    assert all(request_body["return_token_ids"] is True for request_body in captured_bodies)
    assert all(request_body["min_tokens"] == 8 for request_body in captured_bodies)
    assert all(request_body["max_tokens"] == 8 for request_body in captured_bodies)
    assert all(request_body["ignore_eos"] is True for request_body in captured_bodies)
    assert all(request_body["temperature"] == 0.0 for request_body in captured_bodies)
    assert all(request_body["top_p"] == 1.0 for request_body in captured_bodies)
    assert body["observations"][0]["mtp_metrics_before"]["state"] == "unavailable"
    assert body["observations"][0]["mtp_metrics_delta"]["state"] == "unavailable"


def test_bench_command_records_mtp_metric_deltas(monkeypatch, tmp_path):
    metrics_values = iter([0, 8])
    clock = _install_fake_clock(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            drafted = next(metrics_values)
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    observation = json.loads(out.read_text())["observations"][0]
    assert observation["mtp_metrics_before"]["drafted_tokens_total"] == 0.0
    assert observation["mtp_metrics_after"]["drafted_tokens_total"] == 8.0
    assert observation["mtp_metrics_delta"]["state"] == "active"
    assert observation["mtp_metrics_delta"]["drafted_tokens_delta"] == 8.0
    assert observation["deterministic_parity"] is True
    assert observation["parity_basis"] == "token"
    assert observation["parity_failure"] is False


def test_measure_does_not_invent_token_latency_for_multi_token_chunks(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.e2e_output_tokens_per_second == 10.0
    assert result.ttft_ms is None
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_leaves_throughput_unknown_when_usage_and_tokenizer_are_missing(
    monkeypatch,
):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(404, json={"error": "missing"})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens is None
    assert result.e2e_output_tokens_per_second is None
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_uses_tokenizer_count_without_inventing_token_latency(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]
    tokenize_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            tokenize_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"count": 8, "tokens": list(range(8))})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens == 8
    assert result.e2e_output_tokens_per_second == 10.0
    assert tokenize_bodies == [
        {"model": "gemma-4-31b-mtp", "prompt": "abcdefgh"}
    ]
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_measure_uses_streamed_token_ids_for_tpot_and_raw_parity(monkeypatch):
    times = [0.0, 0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"reasoning":"think",'
            '"token_ids":[10,11]}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"ok",'
            '"token_ids":[12,13]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "ok"
    assert result.reasoning_text == "think"
    assert result.raw_output_token_ids == [10, 11, 12, 13]
    assert result.timing_evidence_valid is True
    assert result.token_count_validation_status == "matched"
    assert result.ttft_ms == pytest.approx(100.0)
    assert result.tpot_ms == pytest.approx((0.2 - 0.1) * 1000.0 / 3)
    assert result.tpot_basis == "chunk_arrival_approximation"
    assert result.itl_basis == "not_reported_chunk_interval_only"
    assert result.inter_token_latency_ms_p50 is None


def test_measure_records_sanitized_stream_events_and_separate_ttfts(monkeypatch):
    times_ns = [
        0,
        50_000_000,
        150_000_000,
        250_000_000,
        300_000_000,
        350_000_000,
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"reasoning":"think",'
            '"reasoning_content":"compat","token_ids":[101]}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Hello",'
            '"token_ids":[102,103]}}]}\n\n'
            'data: {"choices":[{"delta":{"reasoning":" more",'
            '"content":" world","token_ids":[104]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._now_ns",
        lambda: times_ns.pop(0) if times_ns else 350_000_000,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "Hello world"
    assert result.reasoning_text == "think more"
    assert result.visible_content == "Hello world"
    assert result.raw_output_token_ids == [101, 102, 103, 104]
    assert result.ttft_ms == pytest.approx(50.0)
    assert result.generated_ttft_ms == pytest.approx(50.0)
    assert result.visible_content_ttft_ms == pytest.approx(150.0)
    assert result.tpot_ms == pytest.approx((0.25 - 0.05) * 1000.0 / 3)
    assert result.tpot_basis == "chunk_arrival_approximation"
    assert result.inter_token_latency_ms_p50 is None
    assert result.itl_basis == "not_reported_chunk_interval_only"
    assert result.chunk_timestamps_ns == [
        50_000_000,
        150_000_000,
        250_000_000,
        300_000_000,
    ]
    assert result.token_chunk_events == [
        {
            "event_index": 1,
            "timestamp_ns": 50_000_000,
            "delta": {
                "token_ids": [101],
                "reasoning": "think",
                "reasoning_content": "compat",
                "content": "",
            },
            "usage": None,
            "finish_reason": None,
            "token_count": 1,
            "multiple_token_ids": False,
            "payload_sha256": result.token_chunk_events[0]["payload_sha256"],
        },
        {
            "event_index": 2,
            "timestamp_ns": 150_000_000,
            "delta": {
                "token_ids": [102, 103],
                "reasoning": "",
                "reasoning_content": "",
                "content": "Hello",
            },
            "usage": None,
            "finish_reason": None,
            "token_count": 2,
            "multiple_token_ids": True,
            "payload_sha256": result.token_chunk_events[1]["payload_sha256"],
        },
        {
            "event_index": 3,
            "timestamp_ns": 250_000_000,
            "delta": {
                "token_ids": [104],
                "reasoning": " more",
                "reasoning_content": "",
                "content": " world",
            },
            "usage": None,
            "finish_reason": None,
            "token_count": 1,
            "multiple_token_ids": False,
            "payload_sha256": result.token_chunk_events[2]["payload_sha256"],
        },
        {
            "event_index": 4,
            "timestamp_ns": 300_000_000,
            "delta": {
                "token_ids": [],
                "reasoning": "",
                "reasoning_content": "",
                "content": "",
            },
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "finish_reason": "stop",
            "token_count": 0,
            "multiple_token_ids": False,
            "payload_sha256": result.token_chunk_events[3]["payload_sha256"],
        },
    ]
    assert result.raw_stream_chunks is None
    assert result.raw_stream_payloads is None
    assert result.raw_stream_capture_status == "disabled"
    assert result.stream_interval_control == "unavailable"


def test_measure_marks_missing_usage_invalid_even_with_raw_token_ids(monkeypatch):
    times = [0.0, 0.1, 0.2]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12,13]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.2,
    )

    _text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert result.raw_output_token_ids == [12, 13]
    assert result.completion_tokens is None
    assert result.e2e_output_tokens_per_second is None
    assert result.timing_evidence_valid is False
    assert result.token_count_validation_status == "usage_missing"
    assert result.token_count_diagnostics == {
        "raw_output_token_count": 2,
        "usage_completion_tokens": None,
        "stream_parse_error_count": 0,
    }
    assert result.ttft_ms is None
    assert result.generated_ttft_ms is None
    assert result.tpot_ms is None


def test_measure_marks_timing_invalid_when_raw_token_count_mismatches_usage(
    monkeypatch,
):
    times = [0.0, 0.0, 0.1, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12,13]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    _text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert result.raw_output_token_ids == [12, 13]
    assert result.completion_tokens == 3
    assert result.timing_evidence_valid is False
    assert result.token_count_validation_status == "usage_mismatch"
    assert result.token_count_diagnostics == {
        "raw_output_token_count": 2,
        "usage_completion_tokens": 3,
        "stream_parse_error_count": 0,
    }
    assert result.ttft_ms is None
    assert result.tpot_ms is None
    assert result.raw_stream_chunks is None
    assert result.raw_stream_payloads is None


def test_measure_invalidates_malformed_sse_without_storing_raw_payloads(monkeypatch):
    times_ns = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12]}}]}\n\n'
            "data: {not-json}\n\n"
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._now_ns",
        lambda: times_ns.pop(0) if times_ns else 400_000_000,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "ok"
    assert result.raw_output_token_ids == [12]
    assert result.timing_evidence_valid is False
    assert result.token_count_validation_status == "malformed_stream"
    assert result.stream_parse_errors == [
        {
            "event_index": 2,
            "timestamp_ns": 200_000_000,
            "error": "JSONDecodeError",
            "payload_sha256": result.stream_parse_errors[0]["payload_sha256"],
            "payload_bytes": len("{not-json}".encode()),
        }
    ]
    assert result.raw_stream_payloads is None
    assert result.raw_stream_chunks is None


def test_measure_raw_stream_capture_is_opt_in_and_secret_scanned(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.3]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12]}}],'
            '"api_key":"sk-proj-secret"}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.3,
    )

    _text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
            capture_raw_stream=True,
        )
    )

    assert result.timing_evidence_valid is True
    assert result.raw_stream_payloads is None
    assert result.raw_stream_chunks is None
    assert result.raw_stream_capture_status == "rejected_by_sanitizer"
    assert result.raw_stream_capture_diagnostics == {
        "rejected_event_indices": [1],
        "reasons": ["secret_pattern"],
    }


def test_measure_does_not_infer_tpot_inside_single_multi_token_chunk(monkeypatch):
    times_ns = [0, 100_000_000, 200_000_000, 300_000_000]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12,13]}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._now_ns",
        lambda: times_ns.pop(0) if times_ns else 300_000_000,
    )

    _text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert result.timing_evidence_valid is True
    assert result.raw_output_token_ids == [12, 13]
    assert result.generated_ttft_ms == pytest.approx(100.0)
    assert result.tpot_ms is None
    assert result.tpot_basis == "unavailable"
    assert result.stream_chunk_interval_ms_p50 == pytest.approx(100.0)


def test_measure_raw_stream_capture_rejects_common_token_key_names(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.3]

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok","token_ids":[12]}}],'
            '"access_token":"opaque-runtime-token"}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.3,
    )

    _text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
            capture_raw_stream=True,
        )
    )

    assert result.raw_stream_payloads is None
    assert result.raw_stream_chunks is None
    assert result.raw_stream_capture_status == "rejected_by_sanitizer"
    assert result.raw_stream_capture_diagnostics == {
        "rejected_event_indices": [1],
        "reasons": ["secret_pattern"],
    }


def test_measure_tokenizer_fallback_uses_visible_text_not_chat_template(monkeypatch):
    times = [0.0, 0.1, 0.2, 0.8]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if "messages" in body:
                return httpx.Response(
                    200,
                    json={"count": 99, "tokens": list(range(99))},
                )
            assert body == {"model": "gemma-4-31b-mtp", "prompt": "abcdefgh"}
            return httpx.Response(200, json={"count": 8})
        body = (
            'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"efgh"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli.time.perf_counter",
        lambda: times.pop(0) if times else 0.8,
    )

    text, result = asyncio.run(
        cli_module._measure(
            "http://mtp:8000",
            {"model": "gemma-4-31b-mtp", "stream": True},
        )
    )

    assert text == "abcdefgh"
    assert result.completion_tokens == 8
    assert result.e2e_output_tokens_per_second == 10.0
    assert result.tpot_ms is None
    assert result.inter_token_latency_ms_p50 is None
    assert result.inter_token_latency_ms_p95 is None


def test_bench_command_writes_v3_artifact_directory(monkeypatch, tmp_path):
    clock = _install_fake_clock(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    "vllm:spec_decode_num_draft_tokens_total 8\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    artifact_root = tmp_path / "artifacts" / "benchmarks"
    runtime_manifest = tmp_path / "runtime-manifest.json"
    runtime_manifest.write_text(
        json.dumps({"argv": ["vllm", "serve"], "vllm_version": "0.21.0"}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "bench",
            "--prompt",
            "hello",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--max-tokens",
            "8",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--artifact-root",
            str(artifact_root),
            "--artifact-id",
            "20260622-test-safe80",
            "--runtime-manifest-path",
            str(runtime_manifest),
        ],
    )

    artifact_dir = artifact_root / "20260622-test-safe80"
    assert result.exit_code == 0, result.stdout
    assert artifact_dir.is_dir()
    expected_files = {
        "manifest.json",
        "results.json",
        "results.md",
        "metrics-before.prom",
        "metrics-after.prom",
        "metrics-delta.json",
        "runtime-manifest.json",
        "request-payloads.json",
        "environment.txt",
        "nvidia-smi.csv",
        "README.md",
    }
    assert expected_files == {path.name for path in artifact_dir.iterdir()}
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    runtime_manifest_payload = json.loads(
        (artifact_dir / "runtime-manifest.json").read_text()
    )
    results = json.loads((artifact_dir / "results.json").read_text())
    request_payloads = json.loads((artifact_dir / "request-payloads.json").read_text())
    serialized = json.dumps(results)

    assert manifest["benchmark_protocol_version"] == 3
    assert manifest["package_version"] == "0.2.0a1"
    assert manifest["runtime_manifest_source"] == "provided"
    assert str(runtime_manifest) not in json.dumps(manifest)
    assert runtime_manifest_payload["vllm_version"] == "0.21.0"
    assert manifest["service_urls"] == {
        "mtp": "http://mtp:8000",
        "baseline": "http://baseline:8000",
    }
    assert request_payloads["chat_completion"]["stream"] is True
    assert results["statistics"]["mtp"]["e2e_output_tokens_per_second"]["median"] == 20.0
    assert "generation_tps" not in serialized


def test_bench_matrix_iterates_prompts_and_n(monkeypatch, tmp_path):
    clock = _install_fake_clock(monkeypatch)
    handler = _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
            "--depth-mtp-url",
            "2=http://mtp2:8000",
            "--depth-mtp-url",
            "4=http://mtp4:8000",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert len(payload) == 4
    keys = {(entry["prompt"], entry["num_speculative_tokens"]) for entry in payload}
    assert keys == {("alpha", 2), ("alpha", 4), ("beta", 2), ("beta", 4)}
    assert all("generation_tps" not in json.dumps(entry) for entry in payload)


def test_bench_matrix_rejects_fake_multi_depth_sweep(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(_make_handler(tps_mtp=20.0, tps_baseline=10.0)),
    )

    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--mtp-url",
            "http://mtp:8000",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
        ],
    )

    assert result.exit_code == 2
    assert "single --mtp-url cannot change live vLLM speculative depth" in result.stderr


def test_bench_matrix_rejects_duplicate_depth_urls(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(_make_handler(tps_mtp=20.0, tps_baseline=10.0)),
    )

    result = runner.invoke(
        app,
        [
            "bench-matrix",
            "--profile",
            "safe80",
            "--baseline-url",
            "http://baseline:8000",
            "--prompt",
            "alpha",
            "--num-speculative-tokens",
            "2",
            "--num-speculative-tokens",
            "4",
            "--depth-mtp-url",
            "2=http://mtp:8000",
            "--depth-mtp-url",
            "4=http://mtp:8000",
        ],
    )

    assert result.exit_code == 2
    assert "distinct --depth-mtp-url endpoint" in result.stderr


def test_bench_single_records_runtime_endpoint_evidence(monkeypatch, tmp_path):
    metrics_values = iter([0, 8])
    clock = _install_fake_clock(monkeypatch)
    captured_bodies: list[dict] = []
    out = tmp_path / "missing" / "nested" / "single.json"

    def handler(request: httpx.Request) -> httpx.Response:
        assert out.parent.is_dir()
        if request.url.path == "/metrics":
            drafted = next(metrics_values)
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        return _make_handler(
            tps_mtp=20.0,
            tps_baseline=10.0,
            clock=clock,
            captured_bodies=captured_bodies,
        )(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
            "--label",
            "eager_true",
            "--profile",
            "tp2_2x32_fp8_gpuonly",
            "--prompt",
            "alpha",
            "--output-token-target",
            "8",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    serialized = json.dumps(payload)
    observation = payload["groups"][0]["observations"][0]
    assert payload["benchmark_protocol_version"] == 3
    assert payload["benchmark_kind"] == "single_endpoint_runtime"
    assert payload["status"] == "complete"
    assert payload["label"] == "eager_true"
    assert payload["profile"] == "tp2_2x32_fp8_gpuonly"
    assert payload["groups"][0]["output_token_target"] == 8
    assert payload["groups"][0]["request_body"]["stream"] is True
    assert payload["groups"][0]["request_body"]["min_tokens"] == 8
    assert observation["result"]["e2e_output_tokens_per_second"] == 20.0
    assert observation["output_sha256"]
    assert observation["output_token_ids"] == [ord(char) for char in "response"]
    assert observation["tokenization_status"] == "matched"
    assert observation["parity_ready"] is True
    assert observation["mtp_metrics_delta"]["state"] == "active"
    assert observation["mtp_metrics_delta"]["drafted_tokens_delta"] == 8.0
    assert (
        payload["groups"][0]["statistics"]["e2e_output_tokens_per_second"]["median"]
        == 20.0
    )
    assert payload["groups"][0]["statistics"]["ttft_ms"]["p95"] is not None
    assert "generation_tps" not in serialized
    assert captured_bodies
    assert all(request_body["ignore_eos"] is True for request_body in captured_bodies)


def test_bench_single_reuses_persistent_client_across_targets(monkeypatch, tmp_path):
    clock = _install_fake_clock(monkeypatch)
    network_stream = object()
    transports: list[_TrackingAsyncTransport] = []

    def handler(request: httpx.Request) -> httpx.Response:
        extensions = {
            "http_version": b"HTTP/1.1",
            "network_stream": network_stream,
        }
        if request.url.path == "/metrics":
            return httpx.Response(200, text="", extensions=extensions)
        if request.url.path == "/v1/chat/completions":
            clock.schedule(completion_tokens=8, tps=20.0)
            return _stream_response(text="response", extensions=extensions)
        return httpx.Response(404, extensions=extensions)

    def transport_factory() -> _TrackingAsyncTransport:
        transport = _TrackingAsyncTransport(handler)
        transports.append(transport)
        return transport

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr("gemma4_mtp_vllm.cli._mock_transport", transport_factory)

    out = tmp_path / "single.json"
    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
            "--prompt",
            "alpha",
            "--output-token-target",
            "8",
            "--output-token-target",
            "16",
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(transports) == 1
    assert transports[0].closed is True
    payload = json.loads(out.read_text())
    observations = [
        observation
        for group in payload["groups"]
        for observation in group["observations"]
    ]
    assert len(observations) == 4
    assert {group["output_token_target"] for group in payload["groups"]} == {8, 16}
    for observation in observations:
        metadata = observation["result"]["transport_metadata"]
        assert metadata["http_version"] == "HTTP/1.1"
        assert metadata["timeout"] == {
            "connect": 10.0,
            "read": None,
            "write": 30.0,
            "pool": 30.0,
        }
        assert metadata["connection_reuse_observable"] is True
        assert metadata["connection_reuse_count"] >= 1
        assert metadata["client_reuse_count"] >= 1


def test_benchmark_client_pool_close_attempts_all_clients_after_close_failure(
    monkeypatch,
):
    first = _CloseProbe(failure=RuntimeError("first close failed"))
    second = _CloseProbe()
    created = iter([first, second])
    monkeypatch.setattr(
        cli_module.BenchmarkHttpClient,
        "create",
        staticmethod(lambda _base_url: next(created)),
    )
    pool = cli_module.BenchmarkClientPool()
    pool.client_for("http://first:8000")
    pool.client_for("http://second:8000")

    with pytest.raises(RuntimeError, match="first close failed"):
        asyncio.run(pool.aclose())

    assert first.closed is True
    assert second.closed is True


def test_benchmark_client_pool_reports_multiple_close_failures_without_exception_group(
    monkeypatch,
):
    first = _CloseProbe(failure=RuntimeError("first close failed"))
    second = _CloseProbe(failure=ValueError("second close failed"))
    created = iter([first, second])
    monkeypatch.setattr(
        cli_module.BenchmarkHttpClient,
        "create",
        staticmethod(lambda _base_url: next(created)),
    )
    pool = cli_module.BenchmarkClientPool()
    pool.client_for("http://first:8000")
    pool.client_for("http://second:8000")

    with pytest.raises(Exception) as exc_info:
        asyncio.run(pool.aclose())

    assert type(exc_info.value).__name__ == "BenchmarkClientCloseError"
    assert [str(error) for error in exc_info.value.errors] == [
        "first close failed",
        "second close failed",
    ]
    assert first.closed is True
    assert second.closed is True


def test_benchmark_client_pool_close_attempts_all_clients_after_close_cancellation(
    monkeypatch,
):
    first = _CloseProbe(failure=asyncio.CancelledError())
    second = _CloseProbe()
    created = iter([first, second])
    monkeypatch.setattr(
        cli_module.BenchmarkHttpClient,
        "create",
        staticmethod(lambda _base_url: next(created)),
    )
    pool = cli_module.BenchmarkClientPool()
    pool.client_for("http://first:8000")
    pool.client_for("http://second:8000")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pool.aclose())

    assert first.closed is True
    assert second.closed is True


def test_measure_cancellation_closes_transport(monkeypatch):
    transports: list[_TrackingAsyncTransport] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            raise asyncio.CancelledError()
        return httpx.Response(200, text="")

    def transport_factory() -> _TrackingAsyncTransport:
        transport = _TrackingAsyncTransport(handler)
        transports.append(transport)
        return transport

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr("gemma4_mtp_vllm.cli._mock_transport", transport_factory)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            cli_module._measure(
                "http://mtp-control:8000",
                {"model": "gemma-4-31b-mtp", "stream": True},
            )
        )

    assert len(transports) == 1
    assert transports[0].closed is True


def test_bench_single_writes_structured_failure_and_closes_client(
    monkeypatch,
    tmp_path,
):
    transports: list[_TrackingAsyncTransport] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        raise httpx.ConnectError("no route", request=request)

    def transport_factory() -> _TrackingAsyncTransport:
        transport = _TrackingAsyncTransport(handler)
        transports.append(transport)
        return transport

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr("gemma4_mtp_vllm.cli._mock_transport", transport_factory)

    out = tmp_path / "failed-single.json"
    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
            "--prompt",
            "alpha",
            "--output-token-target",
            "8",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code != 0
    assert len(transports) == 1
    assert transports[0].closed is True
    payload = json.loads(out.read_text())
    assert payload["status"] == "in_progress"
    assert payload["groups"] == []
    assert payload["failure"] == {
        "kind": "request_failed",
        "phase": "bench-single",
        "url": "http://mtp-control:8000",
        "prompt_name": "prompt_1",
        "output_token_target": 8,
        "exception_type": "ConnectError",
        "message": "no route",
        "request": {
            "method": "POST",
            "url": "http://mtp-control:8000/v1/chat/completions",
        },
        "response": None,
    }


def test_bench_single_flushes_completed_groups_before_later_failure(
    monkeypatch,
    tmp_path,
):
    metrics_values = iter([0, 8, 8])
    clock = _install_fake_clock(monkeypatch)
    out = tmp_path / "partial" / "single.json"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            drafted = next(metrics_values)
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("max_tokens") == 16:
                return httpx.Response(500, json={"error": "boom"})
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
            "--prompt",
            "alpha",
            "--output-token-target",
            "8",
            "--output-token-target",
            "16",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "in_progress"
    assert len(payload["groups"]) == 1
    assert payload["groups"][0]["output_token_target"] == 8
    assert payload["groups"][0]["observations"][0]["parity_ready"] is True
    assert payload["failure"]["exception_type"] == "HTTPStatusError"
    assert payload["failure"]["response"] == {
        "status_code": 500,
        "http_version": "HTTP/1.1",
        "body": '{"error":"boom"}',
    }


def test_bench_single_requires_prompt(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(_make_handler(tps_mtp=20.0, tps_baseline=10.0)),
    )

    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
        ],
    )

    assert result.exit_code == 2
    assert "at least one --prompt required" in result.stderr


def test_bench_single_marks_parity_unready_when_tokenizer_is_unavailable(
    monkeypatch,
    tmp_path,
):
    metrics_values = iter([0, 8])
    clock = _install_fake_clock(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            drafted = next(metrics_values)
            return httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_drafts_total 1\n"
                    f"vllm:spec_decode_num_draft_tokens_total {drafted}\n"
                    "vllm:spec_decode_num_accepted_tokens_total 5\n"
                ),
            )
        if request.url.path == "/v1/chat/completions":
            body = (
                'data: {"choices":[{"delta":{"content":"response"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )
        return _make_handler(tps_mtp=20.0, tps_baseline=10.0, clock=clock)(request)

    monkeypatch.setenv("VLLM_MTP_TRANSPORT_MOCK", "1")
    monkeypatch.setattr(
        "gemma4_mtp_vllm.cli._mock_transport",
        lambda: httpx.MockTransport(handler),
    )

    out = tmp_path / "single.json"
    result = runner.invoke(
        app,
        [
            "bench-single",
            "--url",
            "http://mtp-control:8000",
            "--prompt",
            "alpha",
            "--output-token-target",
            "8",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    observation = json.loads(out.read_text())["groups"][0]["observations"][0]
    assert observation["output_token_ids"] is None
    assert observation["tokenization_status"] == "raw_unavailable"
    assert observation["parity_ready"] is False


def _bench_single_compare_payload(
    *,
    label: str,
    profile: str,
    e2e_values: list[float],
    token_ids: list[int] | None = None,
    parity_ready: bool = True,
    acceptance_rates: list[float] | None = None,
    mean_acceptance_lengths: list[float] | None = None,
    output_token_targets: list[int] | None = None,
    status: str = "complete",
    benchmark_kind: str = "single_endpoint_runtime",
) -> dict[str, object]:
    token_ids = token_ids or [1, 2, 3]
    acceptance_rates = acceptance_rates or [0.60 for _ in e2e_values]
    mean_acceptance_lengths = mean_acceptance_lengths or [3.0 for _ in e2e_values]
    output_token_targets = output_token_targets or [64, 256, 512, 1024]
    groups = []
    for target in output_token_targets:
        observations = []
        for index, e2e in enumerate(e2e_values):
            observations.append(
                {
                    "index": index + 1,
                    "result": {
                        "e2e_output_tokens_per_second": e2e,
                        "ttft_ms": 100.0 + index,
                        "tpot_ms": 20.0 + index,
                    },
                    "output_token_ids": token_ids,
                    "tokenization_status": (
                        "matched" if parity_ready else "raw_unavailable"
                    ),
                    "parity_ready": parity_ready,
                    "mtp_metrics_delta": {
                        "acceptance_rate_delta": acceptance_rates[index],
                        "mean_acceptance_length_delta": mean_acceptance_lengths[index],
                    },
                },
            )
        groups.append(
            {
                "prompt_name": "prompt_1",
                "prompt": "alpha",
                "output_token_target": target,
                "request_body": {"prompt": "alpha", "max_tokens": target},
                "observations": observations,
            }
        )
    return {
        "benchmark_protocol_version": 2,
        "benchmark_kind": benchmark_kind,
        "status": status,
        "label": label,
        "profile": profile,
        "service_url": f"http://{label}:8000",
        "groups": groups,
    }


def test_bench_compare_adopts_candidate_with_complete_evidence(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    out = tmp_path / "nested" / "recommendation.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
                acceptance_rates=[0.60, 0.61],
                mean_acceptance_lengths=[3.0, 3.1],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
                acceptance_rates=[0.61, 0.62],
                mean_acceptance_lengths=[3.1, 3.2],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
            "--same-mode-mtp-parity",
            "passed",
            "--final-answer-quality",
            "passed",
            "--json-output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    group = payload["group_comparisons"][0]
    assert payload["comparison_kind"] == "single_endpoint_runtime_ab"
    assert payload["failure_reasons"] == []
    assert payload["missing_evidence"] == []
    assert payload["recommendation"] == {
        "action": "adopt_candidate",
        "change_default_profile": False,
    }
    assert group["deterministic_parity"] is True
    assert group["e2e_speedup"] > 1.05
    assert group["candidate"]["ttft_ms"]["p95"] is not None
    assert group["candidate"]["mtp_mean_acceptance_length"]["median"] == pytest.approx(
        3.15
    )


def test_bench_compare_requires_external_runtime_evidence(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["failure_reasons"] == []
    assert payload["recommendation"]["action"] == "insufficient_evidence"
    assert payload["recommendation"]["change_default_profile"] is False
    assert payload["missing_evidence"] == [
        "control_startup_seconds_missing",
        "candidate_startup_seconds_missing",
        "control_peak_gpu_memory_mib_missing",
        "candidate_peak_gpu_memory_mib_missing",
        "one_hour_soak_not_passed_or_not_provided",
        "soak_seconds_missing",
        "soak_error_count_missing",
        "no_oom_not_asserted",
        "same_mode_mtp_parity_missing",
        "final_answer_quality_missing",
    ]


def test_bench_compare_treats_cross_mode_parity_failure_as_diagnostic(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
                token_ids=[1, 2, 3],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
                token_ids=[9, 9, 9],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "insufficient_evidence"
    assert "same_mode_mtp_parity_missing" in payload["missing_evidence"]
    assert payload["failure_reasons"] == []
    assert (
        payload["group_comparisons"][0]["parity_reason"]
        == "deterministic_parity_failed"
    )
    assert (
        payload["group_comparisons"][0]["cross_execution_mode_parity"][
            "deterministic_parity"
        ]
        is False
    )


def test_bench_compare_rejects_within_backend_non_repeatability(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_payload = _bench_single_compare_payload(
        label="eager_true",
        profile="tp2_2x32_fp8_gpuonly",
        e2e_values=[100.0, 102.0],
    )
    candidate_payload = _bench_single_compare_payload(
        label="eager_false",
        profile="tp2_2x32_fp8_gpuonly_cuda_graph",
        e2e_values=[110.0, 114.0],
    )
    for group in candidate_payload["groups"]:
        group["observations"][1]["output_token_ids"] = [4, 5, 6]
    control_json.write_text(json.dumps(control_payload), encoding="utf-8")
    candidate_json.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
            "--same-mode-mtp-parity",
            "passed",
            "--final-answer-quality",
            "passed",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "candidate_within_backend_repeatability_failed" in payload[
        "failure_reasons"
    ]


def test_bench_compare_rejects_same_mode_mtp_parity_failure(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
            "--same-mode-mtp-parity",
            "failed",
            "--final-answer-quality",
            "passed",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "same_mode_mtp_parity_failed" in payload["failure_reasons"]


def test_bench_compare_acceptance_non_inferiority_margin(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
                acceptance_rates=[0.60, 0.60],
                mean_acceptance_lengths=[3.0, 3.0],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
                acceptance_rates=[0.595, 0.595],
                mean_acceptance_lengths=[2.96, 2.96],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
            "--same-mode-mtp-parity",
            "passed",
            "--final-answer-quality",
            "passed",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "adopt_candidate"
    assert payload["failure_reasons"] == []
    assert (
        payload["group_comparisons"][0]["acceptance_non_inferiority"]["status"]
        == "passed"
    )


def test_bench_compare_acceptance_non_inferiority_failure(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
                acceptance_rates=[0.60, 0.60],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
                acceptance_rates=[0.58, 0.58],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
            "--same-mode-mtp-parity",
            "passed",
            "--final-answer-quality",
            "passed",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "mtp_acceptance_regression" in payload["failure_reasons"]


def test_bench_compare_rejects_incomplete_target_matrix(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
                output_token_targets=[64],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
                output_token_targets=[64],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "control_missing_required_output_token_target:256" in payload[
        "failure_reasons"
    ]
    assert "candidate_missing_required_output_token_target:1024" in payload[
        "failure_reasons"
    ]


def test_bench_compare_rejects_request_body_and_timing_evidence_gaps(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_payload = _bench_single_compare_payload(
        label="eager_true",
        profile="tp2_2x32_fp8_gpuonly",
        e2e_values=[100.0, 102.0],
    )
    candidate_payload = _bench_single_compare_payload(
        label="eager_false",
        profile="tp2_2x32_fp8_gpuonly_cuda_graph",
        e2e_values=[110.0, 114.0],
    )
    candidate_payload["groups"][0]["request_body"]["temperature"] = 0.7
    for group in candidate_payload["groups"]:
        for observation in group["observations"]:
            observation["result"]["tpot_ms"] = None
    control_json.write_text(json.dumps(control_payload), encoding="utf-8")
    candidate_json.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "request_body_mismatch" in payload["failure_reasons"]
    assert "tpot_evidence_missing" in payload["missing_evidence"]


def test_bench_compare_rejects_invalid_external_evidence(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "-1",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--candidate-peak-gpu-memory-mib",
            "-5",
            "--soak-passed",
            "--soak-seconds",
            "120",
            "--soak-error-count",
            "1",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "control_startup_seconds_invalid" in payload["failure_reasons"]
    assert "candidate_peak_gpu_memory_mib_invalid" in payload["failure_reasons"]
    assert "one_hour_soak_duration_insufficient" in payload["failure_reasons"]
    assert "one_hour_soak_errors_observed" in payload["failure_reasons"]
    assert "control_peak_gpu_memory_mib_per_gpu_incomplete" in payload[
        "missing_evidence"
    ]


def test_bench_compare_rejects_nonfinite_speedup_threshold(tmp_path):
    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(tmp_path / "missing-control.json"),
            "--candidate-json",
            str(tmp_path / "missing-candidate.json"),
            "--min-meaningful-speedup",
            "nan",
        ],
    )

    assert result.exit_code == 2
    assert "--min-meaningful-speedup must be finite and positive" in result.stderr


def test_bench_compare_rejects_nonfinite_external_evidence(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[100.0, 102.0],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "nan",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "inf",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "inf",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "NaN" not in result.stdout
    assert "Infinity" not in result.stdout
    assert payload["control"]["startup_seconds"] is None
    assert payload["candidate"]["peak_gpu_memory_mib"] == [None, 31300.0]
    assert payload["soak"]["duration_seconds"] is None
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "control_startup_seconds_invalid" in payload["failure_reasons"]
    assert "candidate_peak_gpu_memory_mib_invalid" in payload["failure_reasons"]
    assert "soak_seconds_invalid" in payload["failure_reasons"]


def test_bench_compare_rejects_nonfinite_benchmark_metrics(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_payload = _bench_single_compare_payload(
        label="eager_true",
        profile="tp2_2x32_fp8_gpuonly",
        e2e_values=[100.0, 102.0],
    )
    candidate_payload = _bench_single_compare_payload(
        label="eager_false",
        profile="tp2_2x32_fp8_gpuonly_cuda_graph",
        e2e_values=[110.0, 114.0],
    )
    for group in candidate_payload["groups"]:
        for observation in group["observations"]:
            observation["result"]["e2e_output_tokens_per_second"] = float("nan")
            observation["result"]["ttft_ms"] = float("nan")
            observation["result"]["tpot_ms"] = float("inf")
        group["statistics"] = {
            "e2e_output_tokens_per_second": {
                "median": float("nan"),
                "p10": float("nan"),
                "p90": float("inf"),
                "p95": float("inf"),
                "bootstrap_ci_95": {"low": float("nan"), "high": float("inf")},
            },
            "ttft_ms": {"median": float("nan")},
            "tpot_ms": {"median": float("inf")},
        }
    control_json.write_text(json.dumps(control_payload), encoding="utf-8")
    candidate_json.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code != 0
    assert "NaN" not in result.stdout
    assert "Infinity" not in result.stdout
    assert "invalid JSON file" in result.stderr


def test_bench_compare_rejects_nonfinite_pass_through_identity_fields(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        (
            '{"benchmark_protocol_version":2,'
            '"benchmark_kind":"single_endpoint_runtime",'
            '"status":"complete","label":"eager_true","profile":NaN,'
            '"groups":[]}'
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
        ],
    )

    assert result.exit_code != 0
    assert "NaN" not in result.stdout
    assert "invalid JSON file" in result.stderr


def test_bench_compare_rejects_overflowed_json_numbers_cleanly(tmp_path):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        (
            '{"benchmark_protocol_version":2,'
            '"benchmark_kind":"single_endpoint_runtime",'
            '"status":"complete","label":"eager_true","profile":1e999,'
            '"groups":[]}'
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[110.0, 114.0],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
        ],
    )

    assert result.exit_code != 0
    assert result.exit_code != 1
    assert result.stdout == ""
    assert "invalid JSON file" in result.stderr


def test_bench_compare_rejects_overflowed_speedup_without_nonstandard_json(
    tmp_path,
):
    control_json = tmp_path / "control.json"
    candidate_json = tmp_path / "candidate.json"
    control_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_true",
                profile="tp2_2x32_fp8_gpuonly",
                e2e_values=[1e-308, 1e-308],
            )
        ),
        encoding="utf-8",
    )
    candidate_json.write_text(
        json.dumps(
            _bench_single_compare_payload(
                label="eager_false",
                profile="tp2_2x32_fp8_gpuonly_cuda_graph",
                e2e_values=[1e308, 1e308],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bench-compare",
            "--control-json",
            str(control_json),
            "--candidate-json",
            str(candidate_json),
            "--control-startup-seconds",
            "55.0",
            "--candidate-startup-seconds",
            "52.0",
            "--control-peak-gpu-memory-mib",
            "31000",
            "--control-peak-gpu-memory-mib",
            "31100",
            "--candidate-peak-gpu-memory-mib",
            "31200",
            "--candidate-peak-gpu-memory-mib",
            "31300",
            "--soak-passed",
            "--soak-seconds",
            "3600",
            "--soak-error-count",
            "0",
            "--no-oom",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Infinity" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["recommendation"]["action"] == "do_not_adopt"
    assert "meaningful_e2e_speedup_missing" in payload["failure_reasons"]
