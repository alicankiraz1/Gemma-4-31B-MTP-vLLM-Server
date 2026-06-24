# P0-001 Release Baseline

## Scope

Make the current FP8 GPU-only local validation implementation represented by a
clean, versioned Git state before deeper runtime attestation work begins.

## Required Changes

- Bump package metadata to `0.2.0a1`.
- Commit the intended FP8/runtime/profile/source/test/documentation changes.
- Strengthen source archive creation so dirty tracked, staged, and untracked
  files all block release archive generation.
- Strengthen source archive verification so local env files, internal
  superpowers planning files, local absolute paths, and secret-like content are
  rejected.
- Keep runtime artefacts such as `logs/`, `artifacts/`, `dist/`, caches, and
  virtual environments outside the committed source set.

## Verification

- `python -m pytest -q`
- `python -m compileall -q src`
- `python -m pip check`
- `git diff --check`
- source archive build from committed `HEAD`
- source archive verifier negative cases for missing, corrupt, forbidden path,
  local path content, and secret-like content
