# P0-008 P1-001R Code Gate

## Snapshot

- Branch: `codex/p0-008-p1-001r-code-gate`
- Audited base commit before the report-only gate commit: `01dc54a93cc46d2513b40acd4a268b22d0c1f6bf`
- Final branch HEAD: record `git rev-parse HEAD` after the last report/runbook commit
- Final source archive path: `/tmp/gemma4-mtp-src-<final-short-head>.zip`
- Live backend stop/start: not run
- GPU-consuming commands: not run
- Maintenance authorization: still required

## Verification Results

All checks were run from `<repo-root>`.

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | `374 passed, 144 warnings` |
| `.venv/bin/python -m compileall -q src tests` | passed |
| `.venv/bin/python -m pip check` | `No broken requirements found.` |
| `git diff --check` | passed |
| `scripts/verify_wheel_freshness.sh` | built `gemma4_mtp_vllm-0.2.0a1-py3-none-any.whl`; installed-wheel smoke passed |
| `scripts/make_source_archive.sh /tmp/gemma4-mtp-src-<final-short-head>.zip` | must pass after the last report/runbook commit |
| `scripts/verify_source_archive.sh /tmp/gemma4-mtp-src-<final-short-head>.zip` | must pass after the last report/runbook commit |
| `git bundle create /tmp/gemma4-mtp-p0-008-<final-short-head>.bundle HEAD && git bundle verify /tmp/gemma4-mtp-p0-008-<final-short-head>.bundle` | must pass after the last report/runbook commit |

P0-007 reviewer approval at `7d5446786fe2c0347dd7d464d11bbe5d4f7c5357` covered:

- disabled/skipped CUDA graph capture wording in both orders
- vLLM 0.21 `CUDAGraphLogging` multiline table parsing
- mixed log tails with an older disabled line followed by successful capture
- `--vllm-log-path` wiring through `doctor` and `/health`
- no raw log text in response payloads
- Prometheus helper metric filtering
- positive dispatch plus fallback preserving `graph_active=true`

## New CLI Surfaces

- `vllm-mtp doctor --vllm-log-path <path>` reads only the last 256 KiB of the supplied vLLM log and uses it as CUDA graph observation evidence.
- `vllm-mtp serve --vllm-log-path <path>` wires the same evidence into `/health`.
- `bench-single` already records deterministic request bodies and runtime manifests when `--runtime-manifest-path` is supplied.
- `bench-2x2-compare` is the automatic A/B/C/D matrix comparator.
- `bench-compare` remains the B-vs-D performance recommendation comparator with external startup, memory, soak, OOM, parity, and quality gates.

## Maintenance Matrix

Run exactly one backend at a time.

| ID | Profile | Port | MTP | Eager |
| --- | --- | ---: | --- | --- |
| A | `tp2_2x32_fp8_gpuonly` | 8111 | disabled | true |
| B | `tp2_2x32_fp8_gpuonly` | 8112 | enabled | true |
| C | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8113 | disabled | false |
| D | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8114 | enabled | false |

Expected launch argv:

```bash
.venv/bin/vllm-mtp launch --profile tp2_2x32_fp8_gpuonly --port 8111 --no-mtp
.venv/bin/vllm-mtp launch --profile tp2_2x32_fp8_gpuonly --port 8112
.venv/bin/vllm-mtp launch --profile tp2_2x32_fp8_gpuonly_cuda_graph --port 8113 --no-mtp
.venv/bin/vllm-mtp launch --profile tp2_2x32_fp8_gpuonly_cuda_graph --port 8114
```

Expected ports:

- maintenance backends: `8111`, `8112`, `8113`, `8114`
- restored live backend: `$LIVE_BACKEND_PORT`
- restored live gateway: `$LIVE_GATEWAY_PORT`

Evidence root:

```bash
export EVIDENCE=<operator-evidence-root>/<repair-evidence-id>-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$EVIDENCE"/{prestop,matrix/{A,B,C,D},compare,recommendation,rollback}
```

## Pre-Stop Commands

These are read-only and should be captured before stopping anything:

```bash
git rev-parse HEAD | tee "$EVIDENCE/prestop/git-head.txt"
git status --short | tee "$EVIDENCE/prestop/git-status.txt"
ss -H -ltnp "sport = :$LIVE_BACKEND_PORT" | tee "$EVIDENCE/prestop/backend-port.txt" || true
ss -H -ltnp "sport = :$LIVE_GATEWAY_PORT" | tee "$EVIDENCE/prestop/gateway-port.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv | tee "$EVIDENCE/prestop/gpu-processes.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv | tee "$EVIDENCE/prestop/gpu-state.csv"
.venv/bin/vllm-mtp doctor \
  --profile tp2_2x32_fp8_gpuonly \
  --vllm-base-url "$LIVE_BACKEND_URL" \
  --runtime-manifest-path logs/p1-001/live-runtime-manifest.json \
  --vllm-log-path logs/p1-001/live-vllm.log \
  | tee "$EVIDENCE/prestop/live-doctor.json"
```

Abort before the stop gate if any unexpected GPU process is present.

## Per-Backend Evidence

For each matrix entry, write a unique vLLM log and runtime manifest:

```bash
.venv/bin/vllm-mtp launch \
  --profile <PROFILE> \
  --port <PORT> \
  <MTP_FLAG> \
  --manifest-path "$EVIDENCE/matrix/<ID>/runtime-manifest.json" \
  > "$EVIDENCE/matrix/<ID>/vllm.log" 2>&1 &
backend_pid=$!
```

After `/v1/models` is reachable:

```bash
.venv/bin/vllm-mtp doctor \
  --profile <PROFILE> \
  --vllm-base-url http://127.0.0.1:<PORT> \
  --runtime-manifest-path "$EVIDENCE/matrix/<ID>/runtime-manifest.json" \
  --vllm-log-path "$EVIDENCE/matrix/<ID>/vllm.log" \
  | tee "$EVIDENCE/matrix/<ID>/doctor.json"
curl -fsS http://127.0.0.1:<PORT>/metrics > "$EVIDENCE/matrix/<ID>/metrics-before.prom"
```

Benchmark command:

```bash
.venv/bin/vllm-mtp bench-single \
  --url http://127.0.0.1:<PORT> \
  --label <LABEL> \
  --profile <PROFILE> \
  --runtime-manifest-path "$EVIDENCE/matrix/<ID>/runtime-manifest.json" \
  --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
  --output-token-target 64 \
  --output-token-target 256 \
  --output-token-target 512 \
  --output-token-target 1024 \
  --warmup-runs 2 \
  --runs 10 \
  <MTP_FLAG> \
  --json-output "$EVIDENCE/matrix/<ID>/<LABEL>.json"
```

Then capture:

```bash
curl -fsS http://127.0.0.1:<PORT>/metrics > "$EVIDENCE/matrix/<ID>/metrics-after.prom"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$EVIDENCE/matrix/<ID>/gpu-processes-after.csv"
kill "$backend_pid"
wait "$backend_pid" || true
```

## Compare Commands

Automatic 2x2 comparator:

```bash
.venv/bin/vllm-mtp bench-2x2-compare \
  --a-json "$EVIDENCE/matrix/A/eager_no_mtp.json" \
  --b-json "$EVIDENCE/matrix/B/eager_mtp.json" \
  --c-json "$EVIDENCE/matrix/C/graph_no_mtp.json" \
  --d-json "$EVIDENCE/matrix/D/graph_mtp.json" \
  --json-output "$EVIDENCE/compare/p1-001r-2x2.json"
```

B-vs-D recommendation comparator:

Pass `--same-mode-mtp-parity passed` only after
`$EVIDENCE/compare/p1-001r-2x2.json` reports passed same-mode A-vs-B and C-vs-D
gates. If the 2x2 comparator reports missing or failed same-mode parity, keep
the recommendation as `insufficient_evidence` or `do_not_adopt` according to the
comparator failure reason.

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
  --soak-seconds 600 \
  --soak-error-count 0 \
  --no-oom \
  --same-mode-mtp-parity passed \
  --final-answer-quality passed \
  --json-output "$EVIDENCE/recommendation/p1-001r-recommendation.json"
```

## Acceptance And Limitations

- Bootstrap non-inferiority gates remain: acceptance-rate lower CI bound must be `>= -0.01`; mean-acceptance-length lower CI bound must be `>= -0.05`.
- Same-mode parity gates A-vs-B and C-vs-D are correctness gates.
- B-vs-D cross-mode parity is diagnostic, not an automatic `do_not_adopt`.
- Use `--soak-seconds 600` for the new D sanity soak. Use `3600` only when reusing prior soak evidence and the D argv, vLLM, Torch, CUDA, model revisions, and runtime manifest all match.
- CUDA graph observation is evidence-based. `enforce_eager=false` alone never proves graph activity.
- Log evidence requires the operator to pass the correct per-backend `--vllm-log-path`.
- The final recommendation remains `insufficient_evidence` if startup, memory, soak, OOM, parity, quality, runtime manifest, or CUDA graph evidence is missing.

## Rollback

After D and its sanity soak:

```bash
kill "$backend_pid"
wait "$backend_pid" || true
ss -H -ltnp 'sport = :8111' || true
ss -H -ltnp 'sport = :8112' || true
ss -H -ltnp 'sport = :8113' || true
ss -H -ltnp 'sport = :8114' || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv | tee "$EVIDENCE/rollback/gpu-processes-after-matrix.csv"
```

Restore live services with the existing default profile:

```bash
.venv/bin/vllm-mtp launch \
  --profile tp2_2x32_fp8_gpuonly \
  --port "$LIVE_BACKEND_PORT" \
  --manifest-path logs/p1-001/live-runtime-manifest.json \
  > logs/p1-001/live-vllm.log 2>&1 &

.venv/bin/vllm-mtp serve \
  --profile tp2_2x32_fp8_gpuonly \
  --host 127.0.0.1 \
  --port "$LIVE_GATEWAY_PORT" \
  --vllm-base-url "$LIVE_BACKEND_URL" \
  --runtime-manifest-path logs/p1-001/live-runtime-manifest.json \
  --vllm-log-path logs/p1-001/live-vllm.log \
  > logs/p1-001/live-gateway.log 2>&1 &
gateway_pid=$!
printf '%s\n' "$gateway_pid" > "$EVIDENCE/rollback/live-gateway.pid"
```

Post-restore validation must include:

- `vllm-mtp doctor` against `$LIVE_BACKEND_URL`
- gateway `/health`
- OpenAI chat and streaming
- Anthropic messages and streaming
- `/metrics`
- MTP activity after one generation
- live PIDs and ports
- `change_default_profile: false`

## Operator Gate

Do not proceed to stop live gateway/backend or run the A/B/C/D matrix until the
operator explicitly authorizes the maintenance window.
