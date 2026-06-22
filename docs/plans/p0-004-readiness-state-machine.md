# P0-004 Readiness State Machine

## Scope

Add an explicit readiness decision for the gateway without changing the live
vLLM runtime, profiles, or benchmark defaults.

## Design

- Introduce a small readiness evaluator that consumes already observed state:
  vLLM health/version, target model presence, field-level config verification,
  MTP metric state, and gateway runtime counters.
- Keep `/health` backward-compatible by preserving top-level `status`,
  `target_served`, `version_ok`, `config_verification`, and `mtp`.
- Make `/readyz` use the same readiness decision as `/health` instead of only
  checking backend health and version.
- Treat backend unreachable as `unavailable`; backend reachable but not usable
  as `starting`, `warming`, or `degraded` depending on evidence.
- Required config mismatches degrade readiness. Unknown optional fields do not
  fail readiness. Unknown required fields produce warnings until stronger
  runtime evidence exists.
- Before the first gateway generation, zero MTP counters can be `warming` with a
  warning. After a gateway generation, zero speculative activity degrades an MTP
  profile.

## Acceptance Checks

- Wrong served model degrades `/health` and `/readyz`.
- Old vLLM version degrades `/health` and `/readyz`.
- Required runtime config mismatch degrades readiness.
- Pre-generation idle MTP metrics produce warming or ready-with-warning, not
  unavailable.
- Post-generation idle MTP metrics degrade with an explicit warning.
- `/health` and `/readyz` expose a consistent readiness state, reasons,
  warnings, target served status, version status, config summary, and MTP state.
