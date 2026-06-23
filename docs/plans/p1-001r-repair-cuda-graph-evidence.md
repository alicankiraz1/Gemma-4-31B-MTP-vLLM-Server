# P1-001R Repair CUDA-Graph A/B Evidence

## Scope

Repair the P1-001 CUDA-graph/eager benchmark evidence semantics without changing the
live production profile. The current live eager backend and gateway stay in place
until the harness changes are reviewed and committed.

## Design

- Preserve the original maintenance evidence directory by writing derived indexes
  and analysis into a separate P1-001R evidence directory.
- Request `return_token_ids=true` for streaming benchmark calls and use streamed raw
  token IDs as the parity source. Retokenized visible text remains fallback context
  only and is not raw-token parity evidence.
- Store reasoning text, visible content, raw generated token IDs, raw stream chunks,
  timing basis, and token-count validation status per observation.
- Compute TTFT from the first generated token ID arrival and approximate TPOT from
  first-to-last generated token arrival divided by raw token count. When chunks carry
  multiple token IDs, ITL is not exact and is reported as chunk-arrival based.
- Separate parity semantics:
  - within-backend repeatability is an adoption gate;
  - same-execution-mode MTP vs no-MTP parity is required evidence for adoption;
  - cross-execution-mode eager-vs-graph parity is diagnostic only;
  - final-answer quality remains separate evidence.
- Replace zero-tolerance MTP acceptance checks with non-inferiority margins:
  acceptance rate margin `-0.01`, mean acceptance length margin `-0.05`.

## Current Evidence Note

The existing P1-001 B-vs-D benchmark can be reused as speed/stability evidence only
after it is indexed and analyzed. It cannot prove same-mode MTP parity because it
does not include the no-MTP eager and graph baselines.
