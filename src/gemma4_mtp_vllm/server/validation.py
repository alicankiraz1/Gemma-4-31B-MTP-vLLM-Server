from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any


@dataclass
class RequestValidationError(Exception):
    message: str
    code: str = "invalid_request"
    status_code: int = 400

    def __post_init__(self) -> None:
        super().__init__(self.message)


def validate_openai_chat_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _validate_messages(payload.get("messages"), field="messages")
    _optional_positive_int(payload, "max_tokens")
    _optional_positive_int(payload, "top_k")
    _optional_number(payload, "temperature", minimum=0)
    _optional_number(payload, "top_p", minimum=0, maximum=1)
    _optional_bool(payload, "stream")


def validate_openai_completions_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _require_prompt(payload)
    _optional_positive_int(payload, "max_tokens")
    _optional_number(payload, "temperature", minimum=0)
    _optional_number(payload, "top_p", minimum=0, maximum=1)


def validate_anthropic_messages_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _require_positive_int(payload, "max_tokens")
    _validate_messages(payload.get("messages"), field="messages")
    _optional_system(payload)
    _optional_number(payload, "temperature", minimum=0)
    _optional_number(payload, "top_p", minimum=0, maximum=1)
    _optional_positive_int(payload, "top_k")
    _optional_bool(payload, "stream")


def validate_anthropic_count_tokens_payload(payload: dict[str, Any]) -> None:
    _require_model(payload)
    _validate_messages(payload.get("messages"), field="messages")
    _optional_system(payload)


def _require_model(payload: dict[str, Any]) -> None:
    value = payload.get("model")
    if not _is_non_empty_str(value):
        raise RequestValidationError("model must be a non-empty string")


def _require_prompt(payload: dict[str, Any]) -> None:
    value = payload.get("prompt")
    if _is_non_empty_str(value):
        return
    if isinstance(value, list) and value:
        if all(_is_non_empty_str(item) for item in value):
            return
    raise RequestValidationError("prompt must be a non-empty string or list of strings")


def _require_positive_int(payload: dict[str, Any], field: str) -> None:
    if field not in payload:
        raise RequestValidationError(f"{field} is required")
    _validate_positive_int(payload[field], field)


def _optional_positive_int(payload: dict[str, Any], field: str) -> None:
    if field not in payload:
        return
    _validate_positive_int(payload[field], field)


def _validate_positive_int(value: Any, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise RequestValidationError(f"{field} must be a positive integer")


def _optional_number(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if field not in payload:
        return
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise RequestValidationError(f"{field} must be a number")
    if minimum is not None and value < minimum:
        raise RequestValidationError(f"{field} must be at least {minimum:g}")
    if maximum is not None and value > maximum:
        raise RequestValidationError(f"{field} must be at most {maximum:g}")


def _optional_bool(payload: dict[str, Any], field: str) -> None:
    if field not in payload:
        return
    if type(payload[field]) is not bool:
        raise RequestValidationError(f"{field} must be a boolean")


def _optional_system(payload: dict[str, Any]) -> None:
    if "system" not in payload:
        return
    if not _content_is_text(payload["system"]):
        raise RequestValidationError("system must be text-only content")


def _content_is_text(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return all(_is_text_block(block) for block in content)


def _validate_messages(value: Any, *, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise RequestValidationError(f"{field} must be a non-empty list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise RequestValidationError(f"{field}[{index}] must be an object")
        if not _is_non_empty_str(message.get("role")):
            raise RequestValidationError(f"{field}[{index}].role must be a non-empty string")
        if "content" not in message or not _content_is_text(message["content"]):
            raise RequestValidationError(f"{field}[{index}].content must be text-only content")


def _is_text_block(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "text"
        and isinstance(value.get("text"), str)
    )


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""
