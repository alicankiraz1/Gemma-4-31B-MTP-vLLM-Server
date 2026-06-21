from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerLimits:
    max_body_bytes: int = 2 * 1024 * 1024
    max_output_tokens: int = 4096
    max_queue_size: int = 8
    rate_limit_rpm: int = 30
    metrics_enabled: bool = True
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    generation_timeout_seconds: float | None = 900.0

    def __post_init__(self) -> None:
        if self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.rate_limit_rpm < 0:
            raise ValueError("rate_limit_rpm must be non-negative")
        if (
            self.generation_timeout_seconds is not None
            and self.generation_timeout_seconds <= 0
        ):
            raise ValueError("generation_timeout_seconds must be positive")

    def public_dict(self) -> dict[str, int | float | None]:
        return {
            "max_body_bytes": self.max_body_bytes,
            "max_output_tokens": self.max_output_tokens,
            "max_queue_size": self.max_queue_size,
            "rate_limit_rpm": self.rate_limit_rpm,
            "generation_timeout_seconds": self.generation_timeout_seconds,
        }
