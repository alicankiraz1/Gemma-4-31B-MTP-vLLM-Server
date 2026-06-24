# P0-001 Repository Evidence Audit

## Scope

This audit verifies repository state and existing evidence before any further
harness changes. The live backend and gateway were not stopped or restarted, no
GPU-consuming backend was launched, and the two operator-designated evidence
roots were read only.

The sanitized machine-readable index is in
`docs/plans/p0-001-evidence-audit.json`. It contains only SHA256 hashes, byte
counts, relative file names, scan counts, and token/visible-answer diagnostics.
It does not include raw file contents, raw visible answers, command snippets, or
absolute evidence-root paths.

## Repository State

- Audit branch before P0-001 edits: `codex/p0-001-repository-evidence-audit`
- Starting HEAD: `e45ccea6ceff54dc517262b421139ed0c1c3537a`
- `origin/main`: `6eddbdaa11a4cfa540b074aa6cb49afa2265d60c`
- Starting worktree status: clean
- Recent ancestry includes:
  - `e45ccea` Document P1-001R 2x2 maintenance run
  - `a8b55d4` Repair CUDA graph evidence semantics
  - `e881fe4` Prepare CUDA graph eager A/B harness
  - `8ec582c` Scope benchmark docs and release artifacts

P1-001R repair is present in the current branch. The branch contains both
`e881fe4` and `a8b55d4`; the current CLI requests `return_token_ids: true`,
stores `raw_output_token_ids`, labels same-mode MTP parity separately, and keeps
B-vs-D eager-vs-graph parity as cross-mode diagnostic evidence.

## Evidence Index Summary

Generated with the read-only P0-001 audit helper over the two evidence roots.
The full per-file indexes are recorded in the JSON artifact.

| Label | Root name | Files | Dirs | Bytes | Index SHA256 | Secret scan | Local-path scan |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `legacy` | `p1-001-maintenance-20260622T202831Z` | 175 | 13 | 4751306 | `2f61b0f7292c018d8012b3b87ba964d0040bcbe93825841fa8fd971c5a908171` | 0 matches | 49 matches in 26 files |
| `repair` | `p1-001r-repair-20260623T005647Z` | 223 | 18 | 76287789 | `925cbf6b4aa5252d98f11f0aa135abfcd52c2ca11182a43710c69a29b6699544` | 0 matches | 56 matches in 14 files |

Local-path matches are in the raw evidence files only. The committed JSON
records counts and relative file names, not snippets or absolute matched paths.
No files were skipped during indexing.

Committed audit JSON:

- Size: 110255 bytes
- SHA256: `b24548caf84dec310d0f935a797999a08f656f8d0770ea7a43fe6f565ed9f3b3`

## Previous B-vs-D Classification

The legacy P1-001 B-vs-D JSONs did not request `return_token_ids` and did not
store `raw_output_token_ids`. Their `output_token_ids` came from re-tokenizing
visible output through the older harness. Therefore the legacy B-vs-D comparison
is **not raw generation parity**. It is labeled as
`retokenized_visible_output_cross_mode_diagnostic`.

| Target | Control repeatability | Candidate repeatability | LCP tokens | First divergence | Divergence token IDs | Same-position match | Output length equal | Normalized visible-answer similarity |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| 64 | passed | passed | 35 | 35 | `3629` vs `506` | 0.59375 | true | unavailable; visible content not preserved |
| 256 | passed | passed | 35 | 35 | `3629` vs `506` | 0.15234375 | true | unavailable; visible content not preserved |
| 512 | passed | passed | 35 | 35 | `3629` vs `506` | 0.080078125 | true | unavailable; visible content not preserved |
| 1024 | passed | passed | 35 | 35 | `3629` vs `506` | 0.1279296875 | false | unavailable; visible content not preserved |

The repair-run B-vs-D JSONs did request raw stream token IDs and are labeled as
`raw_token_ids_cross_mode_diagnostic`, still diagnostic only because B and D are
different execution modes.

| Target | Control repeatability | Candidate repeatability | LCP tokens | First divergence | Divergence token IDs | Same-position match | Output length equal | Normalized visible-answer similarity |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| 64 | passed | passed | 10 | 10 | `2558` vs `15603` | 0.15625 | true | 0.948170731707317 |
| 256 | passed | passed | 10 | 10 | `2558` vs `15603` | 0.05859375 | true | 0.7604166666666666 |
| 512 | passed | passed | 10 | 10 | `2558` vs `15603` | 0.03515625 | true | 0.4975191700496166 |
| 1024 | passed | passed | 10 | 10 | `2558` vs `15603` | 0.0537109375 | true | 0.49808429118773945 |

## P0-001 Conclusion

- Existing evidence was indexed without modifying evidence roots.
- Secret scan found no secret-like values in either evidence root.
- Raw local paths exist in the remote evidence content, so only sanitized counts,
  relative names, and hashes were copied into repository documentation.
- Legacy B-vs-D parity is not raw generation parity.
- Repair B-vs-D raw token evidence remains cross-mode diagnostic, not a same-mode
  adoption gate.
