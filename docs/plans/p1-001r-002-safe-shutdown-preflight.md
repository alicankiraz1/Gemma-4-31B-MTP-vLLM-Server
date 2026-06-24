# P1-001R-002 Safe Shutdown Preflight

## Kapsam

Bu not P1-001R-002 safe shutdown adimi icin operator onayi sonrasi
uygulanacak preflight, stop, dogrulama ve abort kurallarini sabitler. Bu commit
canli gateway/backend durdurmaz, GPU-consuming deney backend'i baslatmaz ve
default/live profili degistirmez.

Bu not P1-001R-001 baseline kanitina baglidir:

- Baseline evidence name: private baseline evidence ID
- Latest pointer: private baseline pointer file
- Backend PID at capture: captured private PID
- Gateway PID at capture: captured private PID
- Active profile: `tp2_2x32_fp8_gpuonly`
- Backend systemd scope: captured private session scope
- Gateway systemd scope: captured private session scope
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

- live backend private loopback port dinlemeye devam eder.
- live gateway private loopback port dinlemeye devam eder.
- GPU'larda yalnizca mevcut live vLLM worker surecleri kalir.
- Stop komutlari, deney backend'leri ve rollback restore komutlari calistirilmaz.

## Onay Sonrasi Preflight

Operator onayi geldikten sonra ilk is yeni bir shutdown evidence alt dizini
olusturmak olmalidir:

```bash
EVIDENCE="$(cat "$PRIVATE_BASELINE_POINTER")"
mkdir -p "$EVIDENCE/shutdown"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE/shutdown/start-at.txt"
```

Stop oncesi PIDs ve portlar yeniden dogrulanir:

```bash
ss -H -ltnp "sport = :$LIVE_BACKEND_PORT" | tee "$EVIDENCE/shutdown/pre-backend-port.txt"
ss -H -ltnp "sport = :$LIVE_GATEWAY_PORT" | tee "$EVIDENCE/shutdown/pre-gateway-port.txt"
ps -p "$LIVE_BACKEND_PID" -o pid=,comm=,stat=,etimes= | tee "$EVIDENCE/shutdown/pre-backend-ps.txt"
ps -p "$LIVE_GATEWAY_PID" -o pid=,comm=,stat=,etimes= | tee "$EVIDENCE/shutdown/pre-gateway-ps.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  | tee "$EVIDENCE/shutdown/pre-gpu-compute-apps.csv"
```

Abort et ve yeni P1-001R-001 baseline al:

- live backend veya gateway portu dinlemiyorsa
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
kill -TERM "$LIVE_GATEWAY_PID"
for _ in $(seq 1 30); do
  ps -p "$LIVE_GATEWAY_PID" >/dev/null || break
  sleep 1
done
ps -p "$LIVE_GATEWAY_PID" >/dev/null && exit 11

kill -TERM "$LIVE_BACKEND_PID"
for _ in $(seq 1 60); do
  ps -p "$LIVE_BACKEND_PID" >/dev/null || break
  sleep 1
done
ps -p "$LIVE_BACKEND_PID" >/dev/null && exit 12
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE/shutdown/stop-issued-at.txt"
```

Eger gateway PID exit olmazsa backend'e gecmeden once hedefli tani yap. Eger
backend PID exit olmazsa deney backend'i baslatma; operatora durumu bildir ve
restore planini kullan.

## Stop Sonrasi Dogrulama

P1-001R-002 pass sayilmasi icin asagidaki kanitlar ayni shutdown evidence
dizininde bulunmalidir:

```bash
ss -H -ltnp "sport = :$LIVE_GATEWAY_PORT" | tee "$EVIDENCE/shutdown/post-gateway-port.txt" || true
ss -H -ltnp "sport = :$LIVE_BACKEND_PORT" | tee "$EVIDENCE/shutdown/post-backend-port.txt" || true
ps -p "$LIVE_GATEWAY_PID" -o pid=,comm=,stat=,etimes= \
  | tee "$EVIDENCE/shutdown/post-gateway-ps.txt" || true
ps -p "$LIVE_BACKEND_PID" -o pid=,comm=,stat=,etimes= \
  | tee "$EVIDENCE/shutdown/post-backend-ps.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
  | tee "$EVIDENCE/shutdown/post-gpu-compute-apps.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv | tee "$EVIDENCE/shutdown/post-gpu-state.csv"
```

Pass kosullari:

- live gateway portu kapali
- live backend portu kapali
- gateway PID yok
- backend PID yok
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
