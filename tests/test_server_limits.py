from __future__ import annotations

import pytest

from gemma4_mtp_vllm.server.limits import ServerLimits


def test_default_limits_have_safe_values():
    limits = ServerLimits()

    assert limits.max_body_bytes == 2 * 1024 * 1024
    assert limits.max_output_tokens == 4096
    assert limits.max_queue_size == 8
    assert limits.rate_limit_rpm == 30
    assert limits.metrics_enabled is True
    assert limits.cors_origins == ()
    assert limits.generation_timeout_seconds == pytest.approx(900.0)


def test_public_dict_exposes_runtime_fields():
    limits = ServerLimits(max_output_tokens=512, max_queue_size=4)
    payload = limits.public_dict()

    assert payload == {
        "max_body_bytes": 2 * 1024 * 1024,
        "max_output_tokens": 512,
        "max_queue_size": 4,
        "rate_limit_rpm": 30,
        "generation_timeout_seconds": 900.0,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_body_bytes": 0},
        {"max_output_tokens": 0},
        {"max_queue_size": 0},
        {"rate_limit_rpm": -1},
        {"generation_timeout_seconds": 0},
    ],
)
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        ServerLimits(**kwargs)
