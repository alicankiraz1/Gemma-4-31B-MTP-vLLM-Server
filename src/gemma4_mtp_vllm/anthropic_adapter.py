from __future__ import annotations

import uuid
from typing import Any, Iterable, Iterator

from gemma4_mtp_vllm.backend.response_parser import (
    ThoughtSanitizer,
    visible_text_for_history,
)


def anthropic_request_to_openai(
    payload: dict[str, Any],
    *,
    openai_model: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": openai_model,
        "messages": _build_messages(payload),
    }
    for field in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
        if field in payload and payload[field] is not None:
            body[field] = payload[field]
    return body


def openai_response_to_anthropic(
    openai_payload: dict[str, Any],
    *,
    anthropic_model: str,
    message_id_prefix: str,
) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    primary = choices[0] if choices else {}
    content_text = _extract_message_content(primary)
    finish_reason = primary.get("finish_reason")
    usage = openai_payload.get("usage") or {}
    return {
        "id": f"{message_id_prefix}_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": _stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def openai_stream_to_anthropic_events(
    openai_chunks: Iterable[dict[str, Any]],
    *,
    anthropic_model: str,
    message_id_prefix: str,
    prompt_tokens: int,
) -> Iterator[dict[str, Any]]:
    response_id = f"{message_id_prefix}_{uuid.uuid4().hex}"
    output_tokens = 0
    stop_reason = "end_turn"
    started_content = False
    sanitizer = ThoughtSanitizer()

    yield {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": anthropic_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    }

    for chunk in openai_chunks:
        if chunk.get("_done"):
            break
        choices = chunk.get("choices") or []
        if not choices:
            continue
        primary = choices[0]
        delta = primary.get("delta") or {}
        content = delta.get("content")
        if content:
            content = sanitizer.feed(content)
        if content:
            if not started_content:
                yield {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                started_content = True
            output_tokens += 1
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            }
        finish_reason = primary.get("finish_reason")
        if finish_reason:
            stop_reason = _stop_reason(finish_reason)

    content = sanitizer.finish()
    if content:
        if not started_content:
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            started_content = True
        output_tokens += 1
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": content},
        }

    if started_content:
        yield {"type": "content_block_stop", "index": 0}

    yield {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield {"type": "message_stop"}


def _build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = _content_to_text(payload.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    raw_messages = payload.get("messages") or []
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = _content_to_text(
                message.get("content"),
                sanitize_thoughts=role == "assistant",
            )
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )
    return messages


def _content_to_text(content: Any, *, sanitize_thoughts: bool = False) -> str:
    if sanitize_thoughts:
        return _sanitized_content_to_text(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    return str(content)


def _sanitized_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return visible_text_for_history(content)
    sanitizer = ThoughtSanitizer()
    if isinstance(content, list):
        outputs: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            piece = sanitizer.feed(str(block.get("text", "")))
            if piece:
                outputs.append(piece)
        tail = sanitizer.finish()
        if tail:
            outputs.append(tail)
        return visible_text_for_history("\n".join(outputs))
    if isinstance(content, dict) and content.get("type") == "text":
        return visible_text_for_history(str(content.get("text", "")))
    return visible_text_for_history(str(content))


def _extract_message_content(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return visible_text_for_history(content)
    return ""


def _stop_reason(openai_finish: str | None) -> str:
    if openai_finish == "length":
        return "max_tokens"
    return "end_turn"
