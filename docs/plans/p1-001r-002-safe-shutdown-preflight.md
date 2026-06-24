# P1-001R-002 Safe Shutdown Preflight

## Kapsam

Bu not P1-001R-002 safe shutdown adimi icin operator onayi sonrasi
uygulanacak preflight, stop, dogrulama ve abort kurallarini sabitler. Bu commit
canli gateway/backend durdurmaz, GPU-consuming deney backend'i baslatmaz ve
default/live profili degistirmez.

Bu not P1-001R-001 baseline kanitina baglidir:

- Baseline evidence name: `p1-001r-repair-20260624T220543Z`
- Latest pointer: `p1-001r-latest-prestop.path`
- Backend PID at capture: `2809519`
- Gateway PID at capture: `2811072`
- Active profile: `tp2_2x32_fp8_gpuonly`
- Backend systemd scope: `session-7041.scope`
- Gateway systemd scope: `session-7041.scope`
- Rollback command artifact: `pre-stop/rollback-commands.redacted.sh`
- Evidence scan status: `clean`

P1-001R-002 gercek shutdown adimi yalnizca su tek operator onayindan sonra
calistirilabilir:

- stop live gateway
- stop live backend
- run A/B/C/D one backend at a time
- run candidate sanity soak
- restore live eager backend
- restore gateway
- execute post-rollback validation

## Onay Oncesi Durum

Bu hazirlik adimi sonunda beklenen durum:

- `127.0.0.1:8012` live backend dinlemeye devam eder.
- `127.0.0.1:18082` gateway dinlemeye devam eder.
- GPU'larda yalnizca mevcut live vLLM worker surecleri kalir.
- Stop komutlari, deney backend'leri ve rollback restore komutlari calistirilmaz.

## Onay Sonrasi Preflight

Operator onayi geldikten sonra ilk is yeni bir shutdown evidence alt dizini
olusturmak olmalidir:

```bash
EVIDENCE="$(cat "$HOME/p1-001r-latest-prestop.path")"
mkdir -p "$EVIDENCE/shutdown"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE/shutdown/start-at.txt"
```

Stop oncesi PIDs ve portlar yeniden dogrulanir:

```bash
ss -H -ltnp 'sport = :8012' | tee "$EVIDENCE/shutdown/pre-ss-8012.txt"
ss -H -ltnp 'sport = :18082' | tee "$EVIDENCE/shutdown/pre-ss-18082.txt"
ps -p 2809519 -o pid=,comm=,stat=,etimes= | tee "$EVIDENCE/shutdown/pre-backend-ps.txt"
ps -p 2811072 -o pid=,comm=,stat=,etimes= | tee "$EVIDENCE/shutdown/pre-gateway-ps.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  | tee "$EVIDENCE/shutdown/pre-gpu-compute-apps.csv"
```

Abort et ve yeni P1-001R-001 baseline al:

- `8012` veya `18082` dinlemiyorsa
- dinleyen PID capture edilen PID ile eslesmiyorsa
- PID komutu beklenen `vllm` / `vllm-mtp` sureci degilse
- beklenmeyen ek GPU compute process varsa
- `pre-stop/rollback-commands.redacted.sh` yoksa veya okunamiyorsa
- `GATEWAY_API_KEY` restore icin operator tarafindan disaridan saglanamiyorsa

`GATEWAY_API_KEY` evidence dosyalarina yazilmaz.

## Stop Sirasi

Stop sirasi operator onayi alindiktan ve preflight gectikten sonra hedefli PID
ile uygulanir. Broad process-kill komutlari kullanilmaz.

```bash
kill -TERM 2811072
for _ in $(seq 1 30); do
  ps -p 2811072 >/dev/null || break
  sleep 1
done
ps -p 2811072 >/dev/null && exit 11

kill -TERM 2809519
for _ in $(seq 1 60); do
  ps -p 2809519 >/dev/null || break
  sleep 1
done
ps -p 2809519 >/dev/null && exit 12
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE/shutdown/stop-issued-at.txt"
```

Eger gateway PID exit olmazsa backend'e gecmeden once hedefli tani yap. Eger
backend PID exit olmazsa deney backend'i baslatma; operatora durumu bildir ve
restore planini kullan.

## Stop Sonrasi Dogrulama

P1-001R-002 pass sayilmasi icin asagidaki kanitlar ayni shutdown evidence
dizininde bulunmalidir:

```bash
ss -H -ltnp 'sport = :18082' | tee "$EVIDENCE/shutdown/post-ss-18082.txt" || true
ss -H -ltnp 'sport = :8012' | tee "$EVIDENCE/shutdown/post-ss-8012.txt" || true
ps -p 2811072 -o pid=,comm=,stat=,etimes= \
  | tee "$EVIDENCE/shutdown/post-gateway-ps.txt" || true
ps -p 2809519 -o pid=,comm=,stat=,etimes= \
  | tee "$EVIDENCE/shutdown/post-backend-ps.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  | tee "$EVIDENCE/shutdown/post-gpu-compute-apps.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv | tee "$EVIDENCE/shutdown/post-gpu-state.csv"
```

Pass kosullari:

- `18082` kapali
- `8012` kapali
- PID `2811072` yok
- PID `2809519` yok
- GPU compute apps listesinde live vLLM worker kalmamis
- GPU memory A/B/C/D icin yeterli headroom gosteriyor

Herhangi bir pass kosulu saglanmazsa P1-001R-003 baslatilmaz.

## Abort Ve Restore

Abort durumunda deneye gecmeden once restore hedefi current eager profildir:
`tp2_2x32_fp8_gpuonly`.

`pre-stop/rollback-commands.redacted.sh` artifact'i stop ve restore komutlarini
tek yerde tutar. Artifact icindeki API key placeholder olarak
`GATEWAY_API_KEY` ister; key evidence'a yazilmaz. Artifact sadece operator
onayi kapsaminda kullanilir.

P1-001R-002 stop adimi PID'leri zaten kapattiysa artifact dosyasini bastan sona
calistirma; artifact'in restore bolumunu kaynak alarak restore-only komutlari
ayri evidence altinda calistir. Aksi halde artifact'in ilk hedefli `kill`
komutlari kapanmis PID nedeniyle restore'a ulasmadan cikabilir.

Restore sonrasi P1-001R-006 kapsamindaki dogrulamalar calismadan sistem saglikli
sayilmaz:

- `doctor`
- gateway `/health`
- gateway `/readyz`
- backend and gateway `/v1/models`
- OpenAI non-stream
- OpenAI stream
- Anthropic non-stream
- Anthropic stream
- `/metrics`
- MTP smoke
- `backend_errors=0`

## P1-001R-003 Handoff

P1-001R-003 sadece P1-001R-002 pass olduktan sonra baslar. A/B/C/D kosusu
sirasinda tek seferde yalnizca bir backend GPU'lari kullanabilir:

| ID | Profile | Port | MTP | Eager |
| --- | --- | ---: | --- | --- |
| A | `tp2_2x32_fp8_gpuonly` | 8111 | disabled | true |
| B | `tp2_2x32_fp8_gpuonly` | 8112 | enabled | true |
| C | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8113 | disabled | false |
| D | `tp2_2x32_fp8_gpuonly_cuda_graph` | 8114 | enabled | false |

P1-001R-003 baslamadan once `docs/plans/p0-008-p1-001r-code-gate.md`
matrisi ve `docs/plans/p1-001r-001-live-baseline.md` baseline kaniti birlikte
kontrol edilir.

## Bu Committe Calistirilmayanlar

- live gateway stop
- live backend stop
- deney backend launch
- A/B/C/D benchmark
- soak
- rollback restore
- default/live profile degisikligi
