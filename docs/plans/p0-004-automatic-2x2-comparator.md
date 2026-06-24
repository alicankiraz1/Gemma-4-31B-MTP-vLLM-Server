# P0-004 Automatic 2x2 Comparator

## Kapsam

P0-004, dort ayri `bench-single` kanit dosyasini otomatik 2x2 correctness
matrisine donusturur:

- A: `enforce_eager=true`, MTP disabled
- B: `enforce_eager=true`, MTP enabled
- C: `enforce_eager=false`, MTP disabled
- D: `enforce_eager=false`, MTP enabled

Yeni komut:

```bash
vllm-mtp bench-2x2-compare \
  --a-json eager-no-mtp.json \
  --b-json eager-mtp.json \
  --c-json graph-no-mtp.json \
  --d-json graph-mtp.json
```

Eski `bench-compare` iki dosyali P1-001 performans comparator'u olarak geriye
donuk kalir. Yeni 2x2 komut manuel `--same-mode-mtp-parity` veya
`--final-answer-quality` bayragi kabul etmez.

## Kanit sozlesmesi

Her input payload, benchmark gruplarina ek olarak top-level `configuration`
veya `launch_manifest` kaniti tasimalidir. Comparator su alanlari normalize eder:

- `profile`
- `enforce_eager`
- `enable_mtp`
- `argv`, varsa

Eksik configuration invalid experiment sayilir. `argv` varsa
`--enforce-eager` ve `--speculative-config` varligi beklenen role gore
dogrulanir; yanlis argv invalid experiment sayilir.

`bench-single` yeni kanit uretirken top-level `configuration` yazar.
MTP durumu `--enable-mtp/--no-mtp` ile acikca kaydedilir. A ve C no-MTP
rolleri `--no-mtp` ile uretilmelidir. `--runtime-manifest-path` verilirse
sanitized `launch_manifest` de payload'a eklenir ve argv/runtime metadata
karsilastirmasinda kullanilir.

## Karsilastirma kurallari

- A/B ve C/D same-execution-mode MTP correctness gate'tir.
- A/C ve B/D cross-execution-mode diagnostic'tir.
- Cross-mode inequality tek basina `do_not_adopt` uretmez.
- Re-tokenized visible text raw token parity yerine gecmez.
- Butun roller ayni prompt/output-token-target/request body matrisini
  tasimalidir.
- A/B yalniz MTP durumunda, C/D yalniz MTP durumunda, A/C yalniz eager modunda,
  B/D yalniz eager modunda farkli olmalidir.

## Rapor

Her grup/pair icin raporlanacak alanlar:

- exact raw-token equality
- output length equality
- longest common prefix
- first divergence index
- same-position token match percentage
- normalized final-text equality
- edit distance
- finish reason equality
- task validator result equality, varsa

Within-backend repeatability A, B, C, D icin ayri raporlanir. Same-mode MTP
parity failure `do_not_adopt` uretir. Eksik raw-token kaniti
`insufficient_evidence` uretir. Experiment sekil bozuklugu `invalid_experiment`
olarak raporlanir.

## Guvenlik

Bu task yalnizca local JSON comparator kodu ve testlerini degistirir. Canli
backend/gateway durdurmaz, GPU-consuming komut calistirmaz, vLLM ortamini
degistirmez.
