# P0-002 Persistent Benchmark Transport

## Scope

Remove HTTP client construction and fresh TCP connection setup noise from
benchmark timing. Benchmark commands should reuse one `httpx.AsyncClient` per
backend URL for warmups, measured iterations, metric probes, tokenizer fallback,
and every output-token target in the same command run.

## Design

- Add a benchmark client pool keyed by backend base URL.
- Create clients with explicit keep-alive pooling limits and the existing vLLM
  timeout profile.
- Keep timing in `_measure` immediately around `client.stream(...)`; the client
  must already exist before TTFT and total-latency timers start.
- Preserve direct `_measure(base_url, body)` compatibility by using a short-lived
  pool only when no benchmark client is supplied.
- Store transport metadata on each measured result:
  - HTTP version from the chat-completion response.
  - Timeout configuration.
  - Pool/keep-alive configuration.
  - Client request count and client reuse count.
  - Connection reuse count only when the response exposes an observable network
    stream identity.
- Keep separate clients for separate backend URL strings.
- Close every client through async context managers so cancellation and request
  failures do not leak transports.

## Failure Evidence

`bench-single` writes completed groups as before. If a later connection or
request failure occurs, it writes a structured `failure` object to the partial
JSON payload before re-raising the exception. The failure object includes phase,
URL, prompt name, target length, exception type/message, request method/URL, and
response status/body when available.

## Acceptance Checks

- Unit tests prove `bench-single` uses one transport across warmups, measured
  runs, and multiple output-token targets.
- Unit tests prove clients close on success and request failure.
- Unit tests prove request failure writes structured evidence.
- Existing benchmark tests continue to pass.
