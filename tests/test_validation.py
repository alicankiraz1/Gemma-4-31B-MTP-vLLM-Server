from __future__ import annotations

import pytest

from gemma4_mtp_vllm.server.validation import (
    RequestValidationError,
    validate_anthropic_count_tokens_payload,
    validate_anthropic_messages_payload,
    validate_openai_chat_payload,
    validate_openai_completions_payload,
)


def _assert_invalid(call, payload: dict) -> RequestValidationError:
    with pytest.raises(RequestValidationError) as exc_info:
        call(payload)
    exc = exc_info.value
    assert exc.code == "invalid_request"
    assert exc.status_code == 400
    return exc


def test_openai_chat_rejects_non_string_model():
    _assert_invalid(
        validate_openai_chat_payload,
        {"model": [], "messages": [{"role": "user", "content": "hi"}]},
    )


def test_openai_chat_rejects_non_list_messages():
    _assert_invalid(
        validate_openai_chat_payload,
        {"model": "gemma-4-31b-mtp", "messages": "abc"},
    )


def test_openai_chat_rejects_string_temperature():
    _assert_invalid(
        validate_openai_chat_payload,
        {
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": "abc",
        },
    )


def test_openai_completions_rejects_bad_max_tokens():
    _assert_invalid(
        validate_openai_completions_payload,
        {"model": "gemma-4-31b-mtp", "prompt": "hi", "max_tokens": "abc"},
    )


@pytest.mark.parametrize("payload", [
    {"model": "claude-gemma-4-31b-mtp", "messages": [{"role": "user", "content": "hi"}]},
    {
        "model": "claude-gemma-4-31b-mtp",
        "max_tokens": "bad",
        "messages": [{"role": "user", "content": "hi"}],
    },
])
def test_anthropic_messages_rejects_missing_or_bad_max_tokens(payload: dict):
    _assert_invalid(validate_anthropic_messages_payload, payload)


def test_valid_text_block_messages_are_accepted():
    validate_openai_chat_payload(
        {
            "model": "gemma-4-31b-mtp",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
    )
    validate_anthropic_messages_payload(
        {
            "model": "claude-gemma-4-31b-mtp",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "system": [{"type": "text", "text": "Be concise."}],
        }
    )


@pytest.mark.parametrize("field", ["max_tokens", "top_k"])
def test_openai_chat_rejects_bool_for_int_fields(field: str):
    _assert_invalid(
        validate_openai_chat_payload,
        {
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            field: True,
        },
    )


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_anthropic_messages_rejects_bool_for_number_fields(field: str):
    _assert_invalid(
        validate_anthropic_messages_payload,
        {
            "model": "claude-gemma-4-31b-mtp",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            field: False,
        },
    )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_openai_chat_rejects_invalid_top_p_range(value: float):
    _assert_invalid(
        validate_openai_chat_payload,
        {
            "model": "gemma-4-31b-mtp",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": value,
        },
    )


def test_anthropic_count_tokens_rejects_non_list_messages():
    _assert_invalid(
        validate_anthropic_count_tokens_payload,
        {"model": "claude-gemma-4-31b-mtp", "messages": "abc"},
    )
