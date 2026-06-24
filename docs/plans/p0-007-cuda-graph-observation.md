# P0-007 CUDA Graph Observation

## Kapsam

P0-007, CUDA graph aktifligini `enforce_eager=false` varsayimindan ayirir.
Harness yalnizca eldeki metrics/log text kanitini parse eder; canli backend,
gateway veya GPU-consuming maintenance calistirmaz.

## Gozlem Alani

Runtime evidence `cuda_graph` altinda su alanlari raporlar:

- `graph_metrics_registered`
- `graph_capture_observed`
- `graph_dispatch_observed`
- `eager_fallback_observed`
- `graph_dispatch_count`
- `graph_capture_duration_seconds`
- `graph_capture_sizes`
- `graph_evidence_status`
- `graph_active`
- `evidence_sources`

## Karar Semantigi

- `graph_active=true` yalnizca capture veya dispatch evidence varsa uretilir.
- `enforce_eager=false` tek basina `graph_active=true` uretmez.
- Metrics/log evidence yoksa status `unavailable`, `graph_active=null` kalir.
- Eager fallback tek basina gozlenirse status `fallback_observed`,
  `graph_active=false`.
- Capture veya dispatch ile birlikte fallback/miss de gozlenirse status
  `observed_with_fallback`, `graph_active=true`.

## Entegrasyon

`vllm-mtp doctor` ve `vllm-mtp serve` opsiyonel `--vllm-log-path` alir. Bu yol
verildiginde sadece dosyanin son 256 KiB bolumu okunur, parser'a log evidence
olarak iletilir ve ham log response icine konmaz.

## Guvenlik

Parser tolerant ve text-only'dir. Raw startup log response icine konmaz;
maintenance run kaniti kendi sanitize artifact surecinde saklanmalidir.
