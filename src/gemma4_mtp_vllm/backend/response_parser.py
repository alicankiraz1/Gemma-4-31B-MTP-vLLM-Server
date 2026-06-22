from __future__ import annotations

import re
from typing import Any

_CHANNEL_THOUGHT = "<|channel>thought"
_CHANNEL_FINAL = "<|channel>final"
_CHANNEL_CLOSE = "<channel|>"
_HTML_START_NAMES = ("thinking", "thought", "think")
_HTML_CLOSES = ("</thinking>", "</thought>", "</think>")
_VISIBLE_MARKERS = (_CHANNEL_THOUGHT, _CHANNEL_FINAL, _CHANNEL_CLOSE)
_HIDDEN_MARKERS = (_CHANNEL_FINAL, _CHANNEL_CLOSE, *_HTML_CLOSES)


def visible_text_for_history(text: str) -> str:
    sanitizer = ThoughtSanitizer()
    cleaned = sanitizer.feed(text) + sanitizer.finish()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class ThoughtSanitizer:
    def __init__(self) -> None:
        self._buffer = ""
        self._hidden = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        output: list[str] = []
        while self._buffer:
            if self._hidden:
                if self._consume_hidden_marker():
                    continue
                marker_index = self._find_next_hidden_marker()
                if marker_index is not None:
                    self._buffer = self._buffer[marker_index:]
                    continue
                if final:
                    self._buffer = ""
                    break
                keep = _safe_suffix_len(self._buffer, _HIDDEN_MARKERS)
                self._buffer = self._buffer[-keep:] if keep else ""
                break

            if self._consume_visible_marker():
                continue
            html_start = self._html_start_at_buffer_start()
            if html_start is not None:
                tag_end = self._buffer.find(">")
                if tag_end < 0 and not final:
                    break
                if tag_end < 0:
                    output.append(self._buffer)
                    self._buffer = ""
                    break
                self._buffer = self._buffer[tag_end + 1:]
                self._hidden = True
                continue

            marker_index = self._find_next_visible_marker()
            if marker_index is not None:
                output.append(self._buffer[:marker_index])
                self._buffer = self._buffer[marker_index:]
                continue

            if final:
                output.append(self._buffer)
                self._buffer = ""
                break
            keep = _safe_suffix_len(
                self._buffer,
                (*_VISIBLE_MARKERS, *[f"<{name}" for name in _HTML_START_NAMES]),
            )
            if keep:
                output.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            else:
                output.append(self._buffer)
                self._buffer = ""
            break
        return "".join(output)

    def _consume_visible_marker(self) -> bool:
        if self._buffer.startswith(_CHANNEL_THOUGHT):
            self._buffer = self._buffer[len(_CHANNEL_THOUGHT):]
            self._hidden = True
            return True
        if self._buffer.startswith(_CHANNEL_FINAL):
            self._buffer = self._buffer[len(_CHANNEL_FINAL):]
            return True
        if self._buffer.startswith(_CHANNEL_CLOSE):
            self._buffer = self._buffer[len(_CHANNEL_CLOSE):]
            return True
        return False

    def _consume_hidden_marker(self) -> bool:
        if self._buffer.startswith(_CHANNEL_FINAL):
            self._buffer = self._buffer[len(_CHANNEL_FINAL):]
            self._hidden = False
            return True
        if self._buffer.startswith(_CHANNEL_CLOSE):
            self._buffer = self._buffer[len(_CHANNEL_CLOSE):]
            self._hidden = False
            return True
        lower = self._buffer.lower()
        for marker in _HTML_CLOSES:
            if lower.startswith(marker):
                self._buffer = self._buffer[len(marker):]
                self._hidden = False
                return True
        return False

    def _find_next_visible_marker(self) -> int | None:
        lower = self._buffer.lower()
        indexes = [
            index
            for marker in _VISIBLE_MARKERS
            if (index := self._buffer.find(marker)) >= 0
        ]
        for name in _HTML_START_NAMES:
            marker = f"<{name}"
            start = lower.find(marker)
            while start >= 0:
                if _html_tag_boundary(self._buffer, start + len(marker)):
                    indexes.append(start)
                    break
                start = lower.find(marker, start + 1)
        return min(indexes) if indexes else None

    def _find_next_hidden_marker(self) -> int | None:
        lower = self._buffer.lower()
        indexes = [
            index
            for marker in (_CHANNEL_FINAL, _CHANNEL_CLOSE)
            if (index := self._buffer.find(marker)) >= 0
        ]
        indexes.extend(
            index
            for marker in _HTML_CLOSES
            if (index := lower.find(marker)) >= 0
        )
        return min(indexes) if indexes else None

    def _html_start_at_buffer_start(self) -> str | None:
        lower = self._buffer.lower()
        for name in _HTML_START_NAMES:
            marker = f"<{name}"
            if lower.startswith(marker) and _html_tag_boundary(
                self._buffer,
                len(marker),
            ):
                return name
        return None


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


def _safe_suffix_len(text: str, markers: tuple[str, ...]) -> int:
    max_len = min(len(text), max(len(marker) for marker in markers) - 1)
    lower = text.lower()
    lowered_markers = tuple(marker.lower() for marker in markers)
    for length in range(max_len, 0, -1):
        suffix = lower[-length:]
        if any(marker.startswith(suffix) for marker in lowered_markers):
            return length
    return 0


def _html_tag_boundary(text: str, index: int) -> bool:
    if index >= len(text):
        return True
    return text[index] in {">", " ", "\t", "\n", "\r"}
