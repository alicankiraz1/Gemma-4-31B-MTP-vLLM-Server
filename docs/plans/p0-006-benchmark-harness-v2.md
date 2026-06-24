# P0-006 Statistical Correctness

## Kapsam

P0-006, `bench-compare` karar mantiginda sirali ayri backend run'larini
liste index'iyle pair etmeyi kaldirir. Control ve candidate run'lari bagimsiz
orneklem kabul edilir.

## Degisiklik

- E2E throughput, TTFT, TPOT, acceptance-rate delta ve mean-acceptance-length
  delta icin bagimsiz bootstrap median-difference CI raporlanir.
- Her metrikte su alanlar tutulur:
  - control summary: median, p10, p90, 95% CI, sample count
  - candidate summary: median, p10, p90, 95% CI, sample count
  - candidate-minus-control effect size: median difference, p10, p90, 95% CI
- Bootstrap seed deterministiktir ve raporda kaydedilir.
- Raw observations oldugu gibi korunur; istatistikler observation dizilerini
  yeniden yazmaz.
- Non-inferiority kararinda tek median veya index-paired diff kullanilmaz.

## Non-Inferiority

- Acceptance-rate margin: `-0.01` absolute.
- Mean-acceptance-length margin: `-0.05` token.
- Fail yalnizca independent bootstrap `candidate_minus_control` 95% CI alt
  siniri margin altina inerse uretilir.
- Kucuk negatif farklar, CI margin altina gecmiyorsa otomatik failure degildir.

## Guvenlik

Bu task yalnizca lokal comparator kodu ve testleri degistirir. Canli backend,
gateway, GPU-consuming maintenance veya vLLM runtime ortami degismez.
