# P1-001 CUDA Graph / Eager A-B Runbook

This runbook is for a sequential maintenance-window experiment. Do not run it
beside the validated live backend on the same two GPUs.

## Preflight

Record the current live state:

```bash
vllm-mtp doctor \
    --profile tp2_2x32_fp8_gpuonly \
    --vllm-base-url http://127.0.0.1:8012

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
    --format=csv
```

Confirm the experiment profiles differ only by `enforce_eager`:

```bash
vllm-mtp launch --profile tp2_2x32_fp8_gpuonly --port 8101 --print-only
vllm-mtp launch --profile tp2_2x32_fp8_gpuonly_cuda_graph --port 8102 --print-only
```

The control command includes `--enforce-eager`; the candidate command does not.

## Operator Stop Gate

Stop here until the operator approves a maintenance window for interrupting the
validated live gateway/backend. Do not launch any experiment backend while the
live backend or another experiment backend is still using the two GPUs.

Before the control run, stop the validated live gateway/backend using the
operator-approved service command, then verify the experiment has exclusive GPU
capacity:

```bash
ss -ltnp | grep -E ':(8012|18082|8101|8102)' || true
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
    --format=csv
```

Continue only after the live service ports are no longer listening and GPU
memory is available for a single two-GPU Gemma 4 backend.

## Control Run

Launch the control backend on an experiment port:

```bash
vllm-mtp launch \
    --profile tp2_2x32_fp8_gpuonly \
    --host 127.0.0.1 \
    --port 8101 \
    --manifest-path logs/p1-001/control-runtime-manifest.json
```

Measure readiness and run benchmark artefacts for 64, 256, 512, and 1024
output-token targets. Store generated artefacts under ignored local directories,
for example `artifacts/p1-001/` and `bench-results/p1-001/`.

```bash
vllm-mtp bench-single \
    --url http://127.0.0.1:8101 \
    --label eager_true \
    --profile tp2_2x32_fp8_gpuonly \
    --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
    --output-token-target 64 \
    --output-token-target 256 \
    --output-token-target 512 \
    --output-token-target 1024 \
    --runs 10 \
    --warmup-runs 2 \
    --json-output bench-results/p1-001/eager-true.json
```

## Control Stop Gate

Stop the control backend before starting the candidate. Verify the control port
is no longer listening and GPU memory has been released:

```bash
ss -ltnp | grep -E ':(8101|8102)' || true
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
    --format=csv
```

Do not continue to the candidate run until the control backend is stopped.

## Candidate Run

Launch the CUDA graph candidate backend on a separate experiment port:

```bash
vllm-mtp launch \
    --profile tp2_2x32_fp8_gpuonly_cuda_graph \
    --host 127.0.0.1 \
    --port 8102 \
    --manifest-path logs/p1-001/cuda-graph-runtime-manifest.json
```

Run the same prompts, output-token targets, warmup count, run count, and MTP
metric collection used for the control.

```bash
vllm-mtp bench-single \
    --url http://127.0.0.1:8102 \
    --label eager_false \
    --profile tp2_2x32_fp8_gpuonly_cuda_graph \
    --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
    --output-token-target 64 \
    --output-token-target 256 \
    --output-token-target 512 \
    --output-token-target 1024 \
    --runs 10 \
    --warmup-runs 2 \
    --json-output bench-results/p1-001/eager-false.json
```

The single-endpoint JSON records per-run token ids and output hashes. Compare
the control and candidate token ids before using their performance deltas as
adoption evidence. Every run used for the recommendation must have
`parity_ready: true`; runs with `tokenization_status: unavailable` are not
sufficient parity evidence.

## Recommendation Compare

Create the evidence-gated recommendation from the two sequential
`bench-single` outputs plus externally recorded startup, memory, soak, and OOM
evidence:

```bash
vllm-mtp bench-compare \
    --control-json bench-results/p1-001/eager-true.json \
    --candidate-json bench-results/p1-001/eager-false.json \
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
    --json-output bench-results/p1-001/eager-ab-recommendation.json
```

Treat `adopt_candidate` as a recommendation only when `failure_reasons` and
`missing_evidence` are both empty. `insufficient_evidence` means the runtime
comparison may be promising, but the evidence bundle is not complete enough to
change the default profile. `change_default_profile` remains `false`; changing
the live default is a separate reviewed task.

The comparison also verifies the P1-001 scope before it can recommend adoption:
the input profiles must be `tp2_2x32_fp8_gpuonly` and
`tp2_2x32_fp8_gpuonly_cuda_graph`, the repository profile definitions must
differ only by `enforce_eager`, the required 64/256/512/1024 targets must be
present, and each comparable control/candidate group must use an identical
request body with TTFT and TPOT evidence.

## Required Evidence

The recommendation must include:

- startup time to `/v1/models` readiness for both profiles
- peak GPU memory per GPU
- TTFT and TPOT p50/p95 where token timing is available
- `e2e_output_tokens_per_second` for 64, 256, 512, and 1024 output-token targets
- MTP acceptance rate and mean acceptance length
- deterministic parity result
- one-hour soak result and error count
- rollback/restart confirmation for the validated live profile

Do not recommend changing the default profile unless the candidate passes every
adoption gate in `docs/plans/p1-001-cuda-graph-eager-ab.md`.

## Rollback Gate

After the candidate run, stop the candidate backend and restart the validated
live `tp2_2x32_fp8_gpuonly` gateway/backend. Re-run `vllm-mtp doctor` and keep
the doctor output with the evidence bundle before reporting results.
