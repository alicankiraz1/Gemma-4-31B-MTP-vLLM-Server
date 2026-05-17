# Gemma 4 31B MTP vLLM Sidecar Gateway

CUDA / ROCm sibling of the MLX gateway. Runs Google Gemma 4 31B with the
Gemma 4 MTP assistant drafter through vLLM, behind a FastAPI sidecar that
adds OpenAI + Anthropic dual-protocol support, auth, rate limiting,
bounded admission, doctor diagnostics, and a reproducible MTP benchmark.

See `docs/specs/2026-05-16-vllm-sidecar-gateway-design.md` for the design.
See `docs/plans/2026-05-16-vllm-sidecar-gateway-implementation-plan.md` for
the implementation plan currently in progress.
