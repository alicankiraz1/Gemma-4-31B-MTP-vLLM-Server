# P0-005 Throughput And Quality Lanes

## Kapsam

P0-005, forced-length decode benchmark kanitini final-answer kalite kanitindan
ayirir.

## Lane 1: throughput_fixed_length

Mevcut `bench`, `bench-single` ve `bench-matrix` akislari throughput lane olarak
etiketlenir. Request govdesi:

- `temperature=0`
- `top_p=1`
- fixed `seed=1`
- `min_tokens=max_tokens`
- `ignore_eos=true`
- output-token targets: 64, 256, 512, 1024

Bu lane decode performansi, same-mode raw-token parity, MTP metrics, TTFT/TPOT,
memory ve graph davranisi icindir. Final-answer quality gate olarak
kullanilamaz.

## Lane 2: quality_natural_eos

Yeni `bench-quality` komutu dogal EOS kalite suite'i calistirir. Request govdesi:

- `temperature=0`
- `top_p=1`
- fixed `seed=1`
- `ignore_eos=false`
- `min_tokens` yok
- realistic `max_tokens`

Suite yerel ve deterministiktir. Gorev kategorileri:

- coding_python_unit: 10 Python gorevi, AST preflight + executable unit-test validator
- coding_systems_static: 5 Rust/systems statik validator gorevi
- patch_apply: allowed-path git-apply compatible patch validator
- bug_fix_tests: allowed-path patch + AST preflight + repository-test style validator
- structured_json: 10 strict JSON parse/schema validator gorevi
- turkish_technical_security: 10 Turkish technical/security rubric gorevi
- deterministic_reasoning: 10 exact/programmatic answer gorevi
- retrieval_context_grounded: 10 context-grounded source check gorevi
- multi_turn_history_sensitive: 5 history-sensitive gorev, thought leakage check

Quality raporu forced-length throughput metriği uretmez. Rapor alanlari:

- exact answer pass rate
- normalized equivalence pass rate
- executable/static validation pass rate
- JSON validity and schema validity
- patch application success
- truncation rate
- refusal rate
- thought leakage rate

## Guvenlik

Bu task yalnizca harness kodu ve lokal testleri degistirir. Canli backend veya
GPU-consuming maintenance calistirmaz. Python validator gorevleri subprocess
oncesi muhafazakar AST guvenlik kontrolunden gecer, patch validator gorevleri
task'in beklenen dosya allowlist'i disina cikamaz, subprocess adimlari timeout'lu
gecici dizinde calisir; raw reasoning saklanmaz.
