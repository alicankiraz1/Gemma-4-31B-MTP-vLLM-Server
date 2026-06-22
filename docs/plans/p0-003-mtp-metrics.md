# P0-003 MTP Metrics Activity

## Scope

Replace metric-name-only MTP detection with parsed vLLM counter state and
per-run benchmark deltas.

## Design

- Parse vLLM Prometheus text into an `mtp` state object.
- Treat real vLLM 0.21 counters as canonical:
  `spec_decode_num_drafts_total`, `spec_decode_num_draft_tokens_total`, and
  `spec_decode_num_accepted_tokens_total`.
- Ignore `_created` and per-position counters for aggregate totals.
- Return `unavailable` when relevant metrics are absent.
- Return `registered_but_idle` for registered zero-valued counters.
- Return `active` only when draft-round or drafted-token counters are positive,
  or when a before/after delta is positive.
- Preserve `mtp_observed` as a compatibility boolean derived from active state.
- Store parsed MTP metrics before/after each benchmark run plus per-run deltas.

## Verification

- Unit tests cover real vLLM 0.21 fixture parsing, idle metrics, unavailable
  metrics, parse errors, and positive deltas.
- Health and doctor tests cover active and idle MTP states.
- Bench CLI tests cover emitted before/after/delta artefacts.
