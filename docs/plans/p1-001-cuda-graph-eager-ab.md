# P1-001 CUDA Graph / Eager A-B

Compare the validated FP8 GPU-only profile against an isolated CUDA graph
experiment profile. Keep every runtime setting identical except
`enforce_eager`.

## Profiles

- Control: `tp2_2x32_fp8_gpuonly`
  - `enforce_eager: true`
- Candidate: `tp2_2x32_fp8_gpuonly_cuda_graph`
  - `enforce_eager: false`

Both profiles keep:

- `tensor_parallel_size: 2`
- `quantization: fp8`
- `cpu_offload_gb: 0`
- `max_model_len: 2048`
- `max_num_seqs: 1`
- `max_num_batched_tokens: 4096`
- `num_speculative_tokens: 4`

## Runtime Safety

Do not start the candidate while the validated live backend is occupying the
GPUs. The 2x32 GB configuration leaves too little memory headroom for a second
Gemma 4 31B backend. Run this as a maintenance-window, sequential A-B:

1. Capture current live doctor, process argv, metrics, and GPU memory.
2. Stop the live gateway/backend only with operator approval.
3. Run the control profile on an isolated experiment port.
4. Run the candidate profile on a different isolated experiment port.
5. Restart the validated live profile and re-run doctor before reporting.

## Measurements

For both control and candidate collect:

- startup time from launch command to `/v1/models` readiness
- peak GPU memory sampled with `nvidia-smi`
- TTFT
- TPOT
- `e2e_output_tokens_per_second`
- MTP drafted/accepted/rejected token deltas
- 64, 256, 512, and 1024 output-token targets
- deterministic parity
- one-hour soak stability

## Adoption Gate

Recommend `enforce_eager: false` only if all are true:

- no correctness or deterministic-parity regression
- no OOM, preemption, or startup instability
- no MTP acceptance regression
- meaningful TTFT, TPOT, or e2e throughput improvement
- one-hour soak passes

The default/live profile must remain unchanged until the evidence bundle
supports the recommendation.

Use `vllm-mtp bench-compare` to combine the sequential `bench-single` JSON
files with externally recorded startup time, peak GPU memory, soak, and OOM
evidence. A candidate can be considered only when the comparison reports no
`failure_reasons`, no `missing_evidence`, and `recommendation.action` is
`adopt_candidate`. The comparison output must still keep
`change_default_profile: false`; default-profile changes require a separate
reviewed task.

The comparison must reject evidence outside the P1-001 scope, including missing 64/256/512/1024 output-token targets, wrong profiles, profile definitions that differ by more than `enforce_eager`, mismatched request bodies, missing TTFT/TPOT evidence, incomplete per-GPU memory samples, sub-hour soak duration, or non-zero soak errors.
