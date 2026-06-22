from __future__ import annotations

import pytest

from gemma4_mtp_vllm.backend.response_parser import (
    ThoughtSanitizer,
    visible_text_for_history,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<|channel>thought\n<channel|>Final answer", "Final answer"),
        (
            "<|channel>thought\nhidden reasoning\n<channel|>Final answer",
            "Final answer",
        ),
        (
            "<|channel>thought\nhidden reasoning\n<channel|>\n"
            "<|channel>final\nFinal answer",
            "Final answer",
        ),
        ("<think>hidden reasoning</think>Final answer", "Final answer"),
        ("<thought>hidden reasoning</thought>Final answer", "Final answer"),
        ("Plain final answer", "Plain final answer"),
    ],
)
def test_visible_text_for_history_strips_canonical_thoughts(raw, expected):
    assert visible_text_for_history(raw) == expected


def test_visible_text_for_history_keeps_final_after_malformed_channel_block():
    raw = "<|channel>thought\nhidden reasoning\n<|channel>final\nFinal answer"

    assert visible_text_for_history(raw) == "Final answer"


def test_thought_sanitizer_handles_markers_split_across_chunks():
    sanitizer = ThoughtSanitizer()
    chunks = [
        "<|chan",
        "nel>thought\nsecret",
        "\n<chan",
        "nel|>Fi",
        "nal answer",
    ]

    output = "".join(sanitizer.feed(chunk) for chunk in chunks)
    output += sanitizer.finish()

    assert output == "Final answer"
    assert "secret" not in output


def test_thought_sanitizer_handles_split_html_think_markers():
    sanitizer = ThoughtSanitizer()
    chunks = ["<thi", "nk>secret</thi", "nk>Fi", "nal answer"]

    output = "".join(sanitizer.feed(chunk) for chunk in chunks)
    output += sanitizer.finish()

    assert output == "Final answer"
    assert "secret" not in output


def test_thought_sanitizer_keeps_plain_split_text():
    sanitizer = ThoughtSanitizer()

    output = sanitizer.feed("Final ") + sanitizer.feed("answer") + sanitizer.finish()

    assert output == "Final answer"
