from __future__ import annotations

import pytest

from gemma4_mtp_vllm.policy import (
    UnsupportedFeature,
    validate_anthropic_request,
    validate_openai_request,
)


def test_openai_minimal_payload_accepted():
    validate_openai_request({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("field", ["tools", "tool_choice", "function_call", "functions"])
def test_openai_rejects_unsupported_fields(field):
    payload = {"messages": [], field: ["something"]}
    with pytest.raises(UnsupportedFeature) as exc:
        validate_openai_request(payload)
    assert exc.value.code == "unsupported_feature"
    assert field in exc.value.message


def test_openai_accepts_noop_defaults():
    validate_openai_request(
        {
            "messages": [],
            "tools": [],
            "tool_choice": "none",
            "functions": [],
            "function_call": "none",
            "stop": None,
            "response_format": {"type": "text"},
        }
    )


def test_openai_rejects_structured_response_format_when_mtp_enabled():
    payload = {
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    with pytest.raises(UnsupportedFeature):
        validate_openai_request(payload, mtp_enabled=True)


def test_openai_accepts_structured_response_format_when_mtp_disabled():
    payload = {
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    validate_openai_request(payload, mtp_enabled=False)


def test_anthropic_minimal_payload_accepted():
    validate_anthropic_request({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize(
    "field",
    ["tools", "tool_choice", "thinking", "mcp", "files", "stop_sequences"],
)
def test_anthropic_rejects_unsupported_fields(field):
    payload = {"messages": [], field: ["thing"]}
    with pytest.raises(UnsupportedFeature) as exc:
        validate_anthropic_request(payload)
    assert field in exc.value.message
