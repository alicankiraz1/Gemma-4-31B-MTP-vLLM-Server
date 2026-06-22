# P0-006 Benchmark Harness v2

## Scope

Replace the legacy `generation_tps` benchmark contract with a streaming,
artifact-backed benchmark protocol whose metric names describe exactly what is
measured.

## Design

- Use `e2e_output_tokens_per_second` for end-to-end HTTP throughput.
- Record latency fields separately:
  - `ttft_ms`
  - `tpot_ms`
  - `inter_token_latency_ms_p50`
  - `inter_token_latency_ms_p95`
  - `total_latency_ms`
- Force deterministic request shape for benchmark runs:
  - `temperature=0`
  - `top_p=1`
  - `max_tokens=target`
  - `min_tokens=target`
  - `ignore_eos=true`
  - `seed` when configured
- Use streaming responses to measure TTFT, TPOT, and inter-token latencies.
- Report TPOT and inter-token latency only when streamed content events can be
  matched one-to-one with backend completion token counts; otherwise leave
  those fields unknown instead of inventing token latency from grouped chunks.
- Keep metric deltas endpoint-local and capture raw before/after snapshots in
  benchmark artifacts.
- Store immutable benchmark output under:
  `artifacts/benchmarks/<timestamp>-<host>-<profile>/`
- Emit at minimum:
  - `manifest.json`
  - `results.json`
  - `results.md`
  - `metrics-before.prom`
  - `metrics-after.prom`
  - `metrics-delta.json`
  - `runtime-manifest.json`
  - `request-payloads.json`
  - `environment.txt`
  - `nvidia-smi.csv`
  - `README.md`
- Preserve the existing guard that rejects fake multi-depth sweeps using one
  MTP URL for multiple speculative depths.
- Reject duplicate `--depth-mtp-url` endpoint values across speculative depths.

## Acceptance Checks

- Legacy output no longer exposes `generation_tps` metric names.
- Streaming measurements include TTFT, TPOT, inter-token latency, total latency,
  prompt tokens, and completion tokens.
- Benchmark summaries include median, p10, p90, and bootstrap 95% confidence
  intervals for throughput.
- Deterministic parity failures are recorded as benchmark failures.
- Artifact paths are deterministic enough for tests and sanitized for release
  archives.
