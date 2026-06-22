# P0-002 Field-Level Runtime Attestation

## Scope

Replace broad `config_matches` truth with source-aware field verification for
doctor and `/health`.

## Design

- Keep `desired_config` as profile intent plus served alias and vLLM minimum.
- Keep `observed_config` as flat runtime observations only.
- Add `config_verification.status` and `config_verification.fields`.
- Treat profile-only values as `unknown`, never `verified`.
- Derive legacy `config_matches` from `config_verification.status == "verified"`.
- Use vLLM APIs for model/version observations.
- Use metrics only for MTP activity hints.
- Use runtime manifest/process argv only when the manifest PID is still active
  and the current process argv matches the manifest argv.
- Redact argv secrets before returning any process evidence.

## Verification

- Unit tests cover verified, mismatch, partial, and unknown summaries.
- Unit tests cover cpu offload mismatch, profile-only quantization, active-PID
  argv association, and argv redaction.
- Doctor and `/health` tests cover the new response shape while preserving
  existing compatibility fields.
