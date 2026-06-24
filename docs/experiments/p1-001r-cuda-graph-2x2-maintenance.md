# P1-001R CUDA Graph 2x2 Maintenance Runbook

This runbook is for the repaired CUDA-graph evidence run. Do not run any command that stops or starts live services before the operator explicitly approves the maintenance window.

## Scope

- Live backend to restore: `127.0.0.1:8012`
- Live gateway to restore: `127.0.0.1:18082`
- Source commit: `a8b55d474bf876f61f9b983d8a151d36633e4be8`
- Existing evidence to preserve: `<operator-evidence-root>/p1-001-maintenance-20260622T202831Z`
- Repair evidence root: `<operator-evidence-root>/p1-001r-repair-20260623T005647Z`
- Live default profile must not change.
- `change_default_profile` must remain `false`.

## Pre-Stop Gate

Before touching the live processes, record:

```bash
git rev-parse HEAD
git status --short
ss -H -ltnp 'sport = :8012' || true
ss -H -ltnp 'sport = :18082' || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv
.venv/bin/vllm-mtp doctor \
  --profile tp2_2x32_fp8_gpuonly \
  --vllm-base-url http://127.0.0.1:8012 \
  --runtime-manifest-path logs/p1-001/live-runtime-manifest.json
```

Abort if any important GPU process other than the live vLLM workers is present.

## Stop Gate

Stop only the captured live gateway PID and backend PID. Do not use `pkill -f`,
reboot, package upgrades, site-packages patches, or profile edits. Verify
`8012`, `18082`, `8111`, `8112`, `8113`, and `8114` are closed before starting
the matrix.

## Matrix

Run exactly one backend at a time:

| ID | Profile | Port | MTP | Eager |
| --- | --- | ---: | --- | --- |
| A | `tp2_2x32_fp8_gpuonly` | 8111 | disabled | true |
| B | `tp2_2x32_fp8_gpuonly` | 8112 | enabled | true |
| C | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8113 | disabled | false |
| D | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8114 | enabled | false |

The generated argv files under `matrix-prep/` must remain byte-equivalent after
removing port, `--enforce-eager`, and `--speculative-config`.

For every backend, collect:

- exact argv and runtime manifest
- startup duration to `/v1/models`
- metrics before and after benchmark
- peak GPU memory per GPU
- CUDA graph metrics and fallback metrics when the runtime exposes them
- benchmark JSON for 64, 256, 512, and 1024 output-token targets

Benchmark command template:

```bash
.venv/bin/vllm-mtp bench-single \
  --url http://127.0.0.1:<PORT> \
  --label <LABEL> \
  --profile <PROFILE> \
  --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
  --output-token-target 64 \
  --output-token-target 256 \
  --output-token-target 512 \
  --output-token-target 1024 \
  --warmup-runs 2 \
  --runs 10 \
  --json-output "$EVIDENCE/matrix/<ID>/<LABEL>.json"
```

The request body must contain `return_token_ids: true`, `temperature: 0`,
`top_p: 1`, and the same seed for every backend.

## Parity And Quality Gates

Evaluate separately:

- within-backend repeatability for A, B, C, and D
- same-mode MTP parity: A vs B and C vs D
- cross-mode parity: B vs D as diagnostic only
- final-answer exact match and normalized text comparison
- first divergence position, longest common prefix, output length equality, and
  same-position matching-token percentage

Do not treat eager-vs-graph raw-token inequality alone as `do_not_adopt`.

## Acceptance Gates

Use confidence-interval non-inferiority, not zero tolerance:

- acceptance-rate margin: `-0.01`
- mean-acceptance-length margin: `-0.05`

Fail only when the bootstrap 95% CI lower bound of candidate-control is below
the configured margin.

## Candidate Sanity Soak

After the repaired benchmark, run a new 10-minute D sanity soak. Reuse the
previous 3600-second soak only if the D argv, vLLM/Torch/CUDA/model revisions,
and runtime manifest match the previous candidate evidence.

## Recommendation

Run `bench-compare` for the B-vs-D performance diagnostic with repaired semantics,
then include same-mode parity and final-answer quality gate results:

```bash
.venv/bin/vllm-mtp bench-compare \
  --control-json "$EVIDENCE/matrix/B/eager_mtp.json" \
  --candidate-json "$EVIDENCE/matrix/D/graph_mtp.json" \
  --control-startup-seconds <seconds> \
  --candidate-startup-seconds <seconds> \
  --control-peak-gpu-memory-mib <gpu0_mib> \
  --control-peak-gpu-memory-mib <gpu1_mib> \
  --candidate-peak-gpu-memory-mib <gpu0_mib> \
  --candidate-peak-gpu-memory-mib <gpu1_mib> \
  --soak-passed \
  --soak-seconds 3600 \
  --soak-error-count 0 \
  --no-oom \
  --same-mode-mtp-parity passed \
  --final-answer-quality passed \
  --json-output "$EVIDENCE/recommendation/p1-001r-recommendation.json"
```

Use `insufficient_evidence` when raw-token timing, same-mode MTP parity, or
quality evidence is missing. Use `do_not_adopt` for same-mode correctness,
repeatability, quality, soak, OOM, CUDA, NCCL, backend-error, or material
acceptance-collapse failures.

## Rollback Gate

After D and the sanity soak, stop D, verify GPU memory is released, restart:

- vLLM backend on `127.0.0.1:8012` with `tp2_2x32_fp8_gpuonly`
- gateway on `127.0.0.1:18082`

Then verify doctor, `/health`, OpenAI chat, OpenAI streaming, Anthropic messages,
Anthropic streaming, `/metrics`, and MTP activity. The final report must include
the live PIDs and ports and must keep `change_default_profile: false`.
