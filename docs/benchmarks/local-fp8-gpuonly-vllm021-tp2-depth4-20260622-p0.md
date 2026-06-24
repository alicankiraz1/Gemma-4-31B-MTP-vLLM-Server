# Benchmark Record: local-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0

This record scopes the public FP8 GPU-only MTP speedup claim for the
0.2.0a1 hardening line. It is a local hardware/configuration result, not a
universal Gemma 4 MTP performance claim.

## Configuration

- Hardware: 2x NVIDIA GeForce RTX 5090, 32 GB each
- Backend: vLLM 0.21.0
- Torch: 2.11.0+cu130
- Profile: `tp2_2x32_fp8_gpuonly`
- Target: `google/gemma-4-31B-it`
- Drafter: `google/gemma-4-31B-it-assistant`
- Tensor parallel size: 2
- Quantization: fp8
- CPU offload: 0 GB
- Max model length: 2048
- Max concurrent sequences: 1
- Max batched tokens: 4096
- Eager mode: enabled
- Speculative method: MTP
- Speculative depth: 4

## Protocol

The benchmark is a direct vLLM endpoint A/B. It compares one MTP-enabled vLLM
process against a separate vLLM process launched without speculative decoding.
It does not measure gateway overhead.

The benchmark request shape uses deterministic decoding:

- `stream=true`
- `stream_options.include_usage=true`
- `temperature=0`
- `top_p=1`
- `min_tokens=max_tokens`
- `ignore_eos=true`
- `seed=1`

The reported throughput metric is `e2e_output_tokens_per_second`, not
`generation_tps`. It includes the full streamed request window, including HTTP,
queueing/prefill/TTFT, and decode time.

## Result

| Output token target | No-MTP baseline | MTP enabled | Local speedup |
| --- | ---: | ---: | ---: |
| 64 | 13.83 tok/s | 47.79 tok/s | 3.46x |
| 256 | 13.88 tok/s | 54.02 tok/s | 3.89x |
| 512 | 13.78 tok/s | 55.03 tok/s | 3.99x |

The 1024-token MTP-only smoke took approximately 19.9 seconds, or about
51.5 output tok/s. Because the paired no-MTP baseline is not recorded here, it
is not a speedup claim.

## Reproduction

Launch the MTP endpoint:

```bash
vllm-mtp launch --profile tp2_2x32_fp8_gpuonly --port 8001
```

Launch the no-MTP baseline endpoint:

```bash
vllm-mtp launch --profile tp2_2x32_fp8_gpuonly --port 8002 --no-mtp
```

Generate a benchmark artefact for the 256-token target:

```bash
vllm-mtp bench \
    --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
    --profile tp2_2x32_fp8_gpuonly \
    --output-token-target 256 \
    --mtp-url http://127.0.0.1:8001 \
    --baseline-url http://127.0.0.1:8002 \
    --runs 10 \
    --warmup-runs 2 \
    --artifact-root artifacts/benchmarks \
    --artifact-id local-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0 \
    --json-output bench-results/local-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0.json
```

Generate the matrix used for the published output-token targets:

```bash
vllm-mtp bench-matrix \
    --profile tp2_2x32_fp8_gpuonly \
    --baseline-url http://127.0.0.1:8002 \
    --mtp-url http://127.0.0.1:8001 \
    --prompt "Summarize the key trade-offs of running Gemma 4 locally." \
    --num-speculative-tokens 4 \
    --output-token-target 64 \
    --output-token-target 256 \
    --output-token-target 512 \
    --runs 10 \
    --warmup-runs 2 \
    --json-output bench-results/local-fp8-gpuonly-vllm021-tp2-depth4-20260622-p0-matrix.json
```

Generated artefact directories and JSON files are local release evidence, not
source files. Verify them for local paths or secrets before sharing.
