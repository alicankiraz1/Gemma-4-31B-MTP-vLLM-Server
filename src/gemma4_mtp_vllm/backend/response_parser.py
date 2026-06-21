from __future__ import annotations

import re
from typing import Any

_THOUGHT_BLOCK_RE = re.compile(
    r"<(think|thinking|thought)\b[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_THOUGHT_RE = re.compile(
    r"<(think|thinking|thought)\b[^>]*>.*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def visible_text_for_history(text: str) -> str:
    cleaned = _THOUGHT_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THOUGHT_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def finish_reason_from_openai(choice: dict[str, Any]) -> str:
    reason = choice.get("finish_reason") or "stop"
    return str(reason)


def usage_from_openai(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
