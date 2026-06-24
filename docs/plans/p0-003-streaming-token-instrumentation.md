# P0-003 Streaming Token Instrumentation

## Kapsam

P0-003, streaming benchmark kanitini token-id temelli hale getirir. Amac:
generated TTFT, visible-content TTFT, approximate TPOT, ham token id dizisi,
reasoning metni, gorunur final metin ve chunk bazli zaman/metadata kanitini
ayri saklamak.

## Runtime dogrulama durumu

- Benchmark istek govdesi `return_token_ids: true` gonderir.
- Bu checkout icinde `.venv/bin/python` ile `vllm` paket metadata'si
  bulunamadi (`vllm_version unavailable`), bu yuzden local olarak vLLM 0.21.0
  destek dogrulamasi yapilmadi.
- Canli kanit uretiminden once operator, kurulu vLLM 0.21.0 endpoint'inde
  `return_token_ids=true` streaming yanitlarinda `delta.token_ids` dondugunu
  dogrulamalidir.
- Stream interval 1 ayari desteklenirse yalnizca runtime destek kanitindan sonra
  launch profile'a eklenmelidir. Bu kod degisikligi launch argv'sini
  degistirmez ve `stream_interval_control = "unavailable"` kaydeder.

## Tasarim

- Her SSE `data:` chunk'i icin sanitized event metadata kaydedilir:
  `timestamp_ns`, `delta.token_ids`, `delta.reasoning`,
  `delta.reasoning_content`, `delta.content`, `usage`, `finish_reason`,
  token sayisi, multi-token chunk flag'i ve payload SHA-256.
- Raw SSE payload capture default kapali kalir. Yalnizca
  `--capture-raw-stream` ile denenir.
- Raw capture secret/PII scan'den gecmezse payload saklanmaz; sadece sanitized
  reject diagnostics kaydedilir.
- `generated_ttft_ms`, ilk ham generated token id chunk'inin varisina gore
  hesaplanir. Geriye donuk `ttft_ms` bu degerin alias'i olarak kalir.
- `visible_content_ttft_ms`, ilk `delta.content` chunk'inin varisina gore
  hesaplanir.
- `tpot_ms`, ilk ve son generated token id chunk varisi arasindan
  `len(raw_output_token_ids) - 1` boleniyle yaklasik hesaplanir.
- `tpot_basis` her zaman exact olmayan `chunk_arrival_approximation` etiketini
  kullanir.
- Per-token ITL raporlanmaz. Multi-token chunk durumunda chunk interval
  summary'si ayri kalir ve `itl_basis = "not_reported_chunk_interval_only"`
  yazilir.

## Gecerlilik kurallari

- `len(raw_output_token_ids) == usage.completion_tokens` ise timing evidence
  valid olur.
- Usage yoksa veya sayim uyusmazsa `timing_evidence_valid = false`; TTFT/TPOT
  alanlari null kalir.
- Malformed SSE gorulurse parse diagnostics saklanir ve timing evidence
  gecersiz olur.
- `bench-single` adoption/parity hazirligi sadece raw token evidence valid ise
  true olur; mismatch/missing/malformed gozlemler gate icin kullanilmaz.

## Test kapsami

Eklenen/korunan testler sunlari kapsar:

- reasoning-only chunk
- content-only chunk
- mixed reasoning/content chunk
- tek chunk icinde birden cok token id
- token id'lerin chunk'lar arasinda bolunmesi
- usage chunk'inin final DONE oncesi gelmesi
- missing usage
- token-count mismatch
- client cancellation
- malformed SSE
- ilk token'in reasoning token olmasi
- ilk visible-content token'in ilk generated token'dan sonra gelmesi
- raw capture'in default kapali olmasi ve secret scan reddi

