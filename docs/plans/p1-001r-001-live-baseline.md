# P1-001R-001 Live Baseline

## Kapsam

Bu not P1-001R bakim kosusu oncesinde alinan read-only live baseline kanitini
ozetler. Canli gateway/backend durdurulmadi, yeni deney backend'i baslatilmadi,
default/live profil degistirilmedi.

## Remote Hazirlik

- Host: `homelander`
- Izole kod checkout: `p1-001r-code-7b1668a/repo`
- Kod HEAD: `7b1668a7a22be62e904010635f41b92a50a5ccb2`
- Final source archive SHA256: `fc3b45defdc9a437d536acdc9303c145333857783f4a10fec0871dd625611137`
- Final bundle SHA256: `3d960bd53d07e777af9f4c299df6970685c59d95fb315b272bbbe0ca65d767aa`
- Wheel SHA256: `a3305cc2aeb46a1920fb6e5555d42563505ca833974f7664192601b6e7deef71`

Izole checkout, final bundle'dan olusturuldu ve bundle verify edildi. Canli
servisin kullandigi eski checkout veya production vLLM sanal ortami
degistirilmedi.

## Evidence

- Evidence name: `p1-001r-repair-20260624T220543Z`
- Latest pointer on Homelander: `p1-001r-latest-prestop.path`
- Code head recorded in evidence: `7b1668a7a22be62e904010635f41b92a50a5ccb2`
- Live repo head recorded in evidence: `e881fe4ee0cdad10b0babb929bc3cd8a9976ca79`
- Evidence SHA256 manifest: `SHA256SUMS`
- Secret/local-path scan: `clean`
- Rollback command artifact: `pre-stop/rollback-commands.redacted.sh`
- Backend metrics artifact: `pre-stop/backend-metrics.prom`
- Gateway metrics artifact: `pre-stop/gateway-metrics.prom`
- Package/runtime artifacts: `pre-stop/package-versions.json` and
  `pre-stop/live-venv-pip-freeze.txt`

`package-versions.json` records the live gateway package snapshot used for the
pre-stop capture, including `gemma4-mtp-vllm==0.1.0`, `fastapi==0.138.0`,
`httpx==0.28.1`, `pydantic==2.13.4`, `starlette==1.3.1`, and
`typer==0.26.7`. The full live virtualenv freeze is preserved separately in the
evidence artifact.

## Live Process Snapshot

| Component | Port | PID | Status |
| --- | ---: | ---: | --- |
| vLLM backend | 8012 | 2809519 | listening |
| gateway | 18082 | 2811072 | listening |

Current active profile:

- gateway profile: `tp2_2x32_fp8_gpuonly`
- backend systemd scope: `session-7041.scope`
- gateway systemd scope: `session-7041.scope`

Recorded live backend argv confirms the rollback profile shape:

- target model `google/gemma-4-31B-it`
- served model `gemma-4-31b-mtp`
- `--tensor-parallel-size 2`
- `--max-model-len 2048`
- `--gpu-memory-utilization 0.95`
- `--cpu-offload-gb 0`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 4096`
- `--enforce-eager`
- `--quantization fp8`
- MTP speculative config with `num_speculative_tokens: 4`

The gateway argv was recorded with the API key redacted.

`rollback-commands.redacted.sh` records the targeted stop and restore commands
that must be used only after operator approval. It contains the captured PIDs,
uses `GATEWAY_API_KEY` as an external required variable, and does not store the
live API key.

## Live Health

- `doctor-live.json`: `ok=true`, `version_ok=true`, `target_served=true`
- Gateway `/health`: `status=ready`, `readiness.state=ready`
- Gateway `/readyz`: `status=ready`, `readiness.state=ready`
- Backend `/v1/models`: served `gemma-4-31b-mtp`, `max_model_len=2048`
- Gateway `/v1/models`: exposes `gemma-4-31b-mtp` and
  `claude-gemma-4-31b-mtp`
- Backend `/metrics`: captured in `pre-stop/backend-metrics.prom`
- Gateway `/metrics`: captured in `pre-stop/gateway-metrics.prom`; gateway
  counters showed `gemma4_mtp_backend_errors 0` and
  `gemma4_mtp_rejected_requests 0` at capture time
- MTP metrics in doctor: `state=active`
- CUDA graph evidence for the live eager profile: `unavailable`, expected
  because live rollback profile uses `enforce_eager=true`

Gateway health still reports several `config_unknown:*` warnings because the
running gateway process is from the older live checkout and does not use the
new P0-007/P0-008 runtime evidence wiring. The new isolated code path's
`doctor` report verifies the live backend directly.

## API Smoke

Short live smoke requests were captured without changing services:

- OpenAI non-stream chat: response text `OK`, 2 completion tokens
- OpenAI streaming chat: SSE evidence captured
- Anthropic messages: response text `OK`, 2 output tokens

No smoke stderr files were non-empty.

## GPU Snapshot

`nvidia-smi` pre-stop snapshot:

| GPU | Memory Used | Memory Free | Utilization | Power | SM Clock | Memory Clock |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31133 MiB | 977 MiB | 88% | 106.07 W | 2842 MHz | 13801 MHz |
| 1 | 31118 MiB | 991 MiB | 98% | 107.86 W | 2865 MHz | 13801 MHz |

Compute apps:

- `VLLM::Worker_TP0`: 31108 MiB
- `VLLM::Worker_TP1`: 31108 MiB

## Devam Kapisi

P1-001R-001 read-only baseline kaniti ve redacted rollback command artifact'i
mevcut. Bir sonraki adim P1-001R-002 safe shutdown ve ardindan A/B/C/D
kosusudur. Bu adimlar canli gateway/backend durdurmayi ve GPU-consuming deney
backend'leri baslatmayi gerektirdigi icin tek operator onayi olmadan devam
edilmemelidir.
