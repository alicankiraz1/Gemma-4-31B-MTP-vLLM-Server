# P0-007 Documentation And Release Artefacts

Make public performance claims reproducible and scoped to immutable evidence.

## Scope

- Replace stale README benchmark badges and tables with a benchmark ID scoped to
  the Homelander FP8 GPU-only direct-vLLM A/B result.
- Document the exact runtime configuration behind the local 3.46x-3.99x
  speedup range.
- Separate BF16 CPU-offload smoke validation from FP8 GPU-only throughput.
- Separate direct-vLLM MTP/no-MTP speedup from gateway-overhead benchmarking.
- Keep release artifact instructions on clean source archives and fresh wheel
  smoke tests.

## Implementation Notes

- Add a public benchmark record under `docs/benchmarks/` so the source archive
  carries the scoped claim and reproduction commands.
- Keep generated benchmark artifacts under ignored `artifacts/` and
  `bench-results/`; do not commit generated local outputs.
- Update `verify_wheel_freshness.sh` so the installed wheel smoke asserts the
  installed package version is `0.2.0a1`.
- Add release-script tests that fail on stale unscoped benchmark values and
  require the immutable benchmark ID.

## Acceptance Checks

- README has no stale 2.12x/62-136 tok/s public claims.
- README and benchmark record state the FP8 GPU-only configuration and local
  scope.
- Release archive verification still rejects local paths, secrets, caches, and
  internal planning files.
- Wheel smoke checks the installed package version.
- Full test, compile, pip, diff, path/secret, wheel, and source archive gates
  pass locally and on Homelander.
