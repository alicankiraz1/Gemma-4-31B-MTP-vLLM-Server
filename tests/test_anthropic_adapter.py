from __future__ import annotations

import json

from gemma4_mtp_vllm.anthropic_adapter import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
)


def test_anthropic_request_translates_system_and_messages():
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "system": "Be concise.",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "4"}],
            },
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    openai_body = anthropic_request_to_openai(
        payload, openai_model="google/gemma-4-31B-it",
    )
    assert openai_body == {
        "model": "google/gemma-4-31B-it",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_anthropic_request_strips_assistant_thoughts_from_history():
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "messages": [
            {"role": "user", "content": "literal <thought>keep</thought>"},
            {
                "role": "assistant",
                "content": "<thought>private chain</thought>\nVisible answer",
            },
        ],
    }
    openai_body = anthropic_request_to_openai(
        payload,
        openai_model="google/gemma-4-31B-it",
    )

    assert openai_body["messages"] == [
        {"role": "user", "content": "literal <thought>keep</thought>"},
        {"role": "assistant", "content": "Visible answer"},
    ]


def test_anthropic_request_strips_assistant_thoughts_split_across_blocks():
    payload = {
        "model": "claude-gemma-4-31b-mtp",
        "messages": [
            {"role": "user", "content": "literal <think>keep</think>"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<thi"},
                    {"type": "text", "text": "nk>secret</think>Visible answer"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<|chan"},
                    {"type": "text", "text": "nel>thought\nsecret"},
                    {"type": "text", "text": "\n<channel|>Final answer"},
                ],
            },
        ],
    }
    openai_body = anthropic_request_to_openai(
        payload,
        openai_model="google/gemma-4-31B-it",
    )

    assert openai_body["messages"] == [
        {"role": "user", "content": "literal <think>keep</think>"},
        {"role": "assistant", "content": "Visible answer"},
        {"role": "assistant", "content": "Final answer"},
    ]
    assert "secret" not in json.dumps(openai_body)


def test_openai_response_to_anthropic_envelope():
    openai_payload = {
        "id": "chatcmpl-abc",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }
    body = openai_response_to_anthropic(
        openai_payload,
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-gemma-4-31b-mtp"
    assert body["content"] == [{"type": "text", "text": "Hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 1}
    assert body["id"].startswith("msg_")


def test_openai_response_to_anthropic_strips_thought_tags():
    body = openai_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>scratchpad</think>\nFinal answer",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )

    assert body["content"] == [{"type": "text", "text": "Final answer"}]


def test_openai_response_to_anthropic_strips_channel_thoughts():
    body = openai_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<|channel>thought\nsecret\n<channel|>"
                            "<|channel>final\nFinal answer"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )

    assert body["content"] == [{"type": "text", "text": "Final answer"}]


def test_openai_response_max_tokens_maps_to_anthropic_stop_reason():
    body = openai_response_to_anthropic(
        {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        anthropic_model="claude-gemma-4-31b-mtp",
        message_id_prefix="msg",
    )
    assert body["stop_reason"] == "max_tokens"


def test_openai_stream_to_anthropic_events_smoke():
    openai_chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"_done": True},
    ]
    events = list(
        openai_stream_to_anthropic_events(
            openai_chunks,
            anthropic_model="claude-gemma-4-31b-mtp",
            message_id_prefix="msg",
            prompt_tokens=1,
        )
    )
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert types.count("content_block_delta") == 2
    assert types[-2:] == ["message_delta", "message_stop"]


def test_openai_stream_to_anthropic_events_strips_split_thought_markers():
    openai_chunks = [
        {"choices": [{"delta": {"content": "<|chan"}}]},
        {"choices": [{"delta": {"content": "nel>thought\nsecret"}}]},
        {"choices": [{"delta": {"content": "\n<chan"}}]},
        {"choices": [{"delta": {"content": "nel|>Fi"}}]},
        {"choices": [{"delta": {"content": "nal answer"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"_done": True},
    ]
    events = list(
        openai_stream_to_anthropic_events(
            openai_chunks,
            anthropic_model="claude-gemma-4-31b-mtp",
            message_id_prefix="msg",
            prompt_tokens=1,
        )
    )
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta"
    )

    assert text == "Final answer"
    assert "secret" not in str(events)
    assert "<|channel>" not in str(events)
