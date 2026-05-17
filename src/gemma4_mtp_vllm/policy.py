from __future__ import annotations

from typing import Any


class UnsupportedFeature(Exception):
    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


_OPENAI_REJECT_FIELDS = ("tools", "tool_choice", "function_call", "functions", "stop")
_ANTHROPIC_REJECT_FIELDS = (
    "tools",
    "tool_choice",
    "thinking",
    "mcp",
    "files",
    "stop_sequences",
)


def validate_openai_request(payload: dict[str, Any], *, mtp_enabled: bool = True) -> None:
    for field in _OPENAI_REJECT_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if _is_openai_noop(field, value):
            continue
        raise UnsupportedFeature(
            status_code=400,
            code="unsupported_feature",
            message=f"openai field {field!r} is not supported in v1",
        )

    response_format = payload.get("response_format")
    if response_format is not None and isinstance(response_format, dict):
        format_type = response_format.get("type")
        if format_type and format_type != "text" and mtp_enabled:
            raise UnsupportedFeature(
                status_code=400,
                code="unsupported_feature",
                message=(
                    "openai field 'response_format' with structured types is "
                    "not supported while mtp is enabled"
                ),
            )


def validate_anthropic_request(payload: dict[str, Any]) -> None:
    for field in _ANTHROPIC_REJECT_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if _is_anthropic_noop(field, value):
            continue
        raise UnsupportedFeature(
            status_code=400,
            code="unsupported_feature",
            message=f"anthropic field {field!r} is not supported in v1",
        )


def _is_openai_noop(field: str, value: Any) -> bool:
    if field == "tools" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "tool_choice" and (value is None or value == "none"):
        return True
    if field == "functions" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "function_call" and (value is None or value == "none"):
        return True
    if field == "stop" and value is None:
        return True
    return False


def _is_anthropic_noop(field: str, value: Any) -> bool:
    if field == "tools" and isinstance(value, list) and len(value) == 0:
        return True
    if field == "tool_choice" and isinstance(value, dict) and value.get("type") == "none":
        return True
    if field == "thinking" and isinstance(value, dict) and value.get("type") == "disabled":
        return True
    if field == "stop_sequences" and isinstance(value, list) and len(value) == 0:
        return True
    if field in {"mcp", "files"} and isinstance(value, list) and len(value) == 0:
        return True
    return False
