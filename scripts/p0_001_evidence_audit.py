#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(rb"hf_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
]
LOCAL_PATH_PATTERNS = [
    re.compile(rb"(?:/Users|/home|/Volumes|/var/folders)(?:/[^\0\r\n\t\"'<>|]+)+"),
    re.compile(
        rb"/(?:tmp|var|opt|mnt|srv|etc|private|Applications|Library|System|"
        rb"workspace|workspaces|data|root)(?:/[^\0\r\n\t\"'<>|]+)+"
    ),
    re.compile(rb"file:///(?:[^\0\r\n\t\"'<>|]+/)+[^\0\r\n\t\"'<>|]+"),
    re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n\t \"']+"),
]
SCAN_CHUNK_BYTES = 1024 * 1024
SCAN_OVERLAP_BYTES = 8192


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build sanitized P0-001 evidence indexes and B-vs-D diagnostics. "
            "The command is read-only: it hashes and scans files but writes only "
            "to stdout."
        )
    )
    parser.add_argument(
        "--evidence-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Evidence directory to index. May be provided more than once.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LABEL=CONTROL_JSON,CANDIDATE_JSON",
        help="bench-single JSON pair to compare. May be provided more than once.",
    )
    parser.add_argument(
        "--omit-file-index",
        action="store_true",
        help="Keep directory-level hashes and scan summaries, but omit per-file rows.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Override generated_at_utc for deterministic audit fixtures.",
    )
    args = parser.parse_args()

    payload = build_audit_payload(
        evidence_roots=[_parse_label_path(item) for item in args.evidence_root],
        pairs=[_parse_pair(item) for item in args.pair],
        include_file_index=not args.omit_file_index,
        generated_at_utc=args.generated_at_utc,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def build_audit_payload(
    *,
    evidence_roots: list[tuple[str, Path]],
    pairs: list[tuple[str, Path, Path]],
    include_file_index: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc
        or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "evidence_roots": [
            build_evidence_root_index(
                label=label,
                root=root,
                include_file_index=include_file_index,
            )
            for label, root in evidence_roots
        ],
        "b_vs_d_pairs": [
            analyze_b_vs_d_pair(label=label, control_path=control, candidate_path=candidate)
            for label, control, candidate in pairs
        ],
    }


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"invalid LABEL=PATH value: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"invalid empty label in: {value}")
    return label, Path(raw_path)


def _parse_pair(value: str) -> tuple[str, Path, Path]:
    if "=" not in value:
        raise SystemExit(f"invalid LABEL=CONTROL_JSON,CANDIDATE_JSON value: {value}")
    label, raw_paths = value.split("=", 1)
    paths = raw_paths.split(",", 1)
    if not label.strip() or len(paths) != 2 or not paths[0] or not paths[1]:
        raise SystemExit(f"invalid pair value: {value}")
    return label.strip(), Path(paths[0]), Path(paths[1])


def build_evidence_root_index(
    *,
    label: str,
    root: Path,
    include_file_index: bool,
) -> dict[str, Any]:
    root = root.resolve()
    file_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    dir_count = 0
    total_bytes = 0
    secret_scan = ScanSummary()
    local_path_scan = ScanSummary()

    base_payload: dict[str, Any] = {
        "label": label,
        "root_name": root.name,
        "file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "index_format": "sha256  size_bytes  relative_path",
        "index_sha256": _sha256_text(""),
        "secret_scan": secret_scan.to_dict(),
        "local_path_scan": local_path_scan.to_dict(),
        "skipped": skipped_rows,
        "errors": error_rows,
    }
    if not root.exists():
        error_rows.append({"relative_path": ".", "reason": "root_missing"})
        base_payload["status"] = "invalid"
        if include_file_index:
            base_payload["files"] = []
        return base_payload
    if not root.is_dir():
        error_rows.append({"relative_path": ".", "reason": "root_not_directory"})
        base_payload["status"] = "invalid"
        if include_file_index:
            base_payload["files"] = []
        return base_payload
    if not os.access(root, os.R_OK | os.X_OK):
        error_rows.append({"relative_path": ".", "reason": "root_inaccessible"})
        base_payload["status"] = "inaccessible"
        if include_file_index:
            base_payload["files"] = []
        return base_payload

    def walk_error(error: OSError) -> None:
        error_rows.append(
            {
                "relative_path": _relative_error_path(root, error.filename),
                "reason": "walk_error",
                "error_type": type(error).__name__,
            }
        )

    for dirpath, dirnames, filenames in os.walk(
        root,
        followlinks=False,
        onerror=walk_error,
    ):
        filtered_dirnames = []
        for dirname in sorted(dirnames):
            child_dir = Path(dirpath) / dirname
            try:
                if child_dir.is_symlink():
                    skipped_rows.append(
                        {
                            "relative_path": _relative_path(root, child_dir),
                            "reason": "symlink_directory",
                        }
                    )
                    continue
            except OSError as exc:
                skipped_rows.append(
                    {
                        "relative_path": _relative_path(root, child_dir),
                        "reason": "directory_stat_error",
                    }
                )
                error_rows.append(
                    {
                        "relative_path": _relative_path(root, child_dir),
                        "reason": "directory_stat_error",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            filtered_dirnames.append(dirname)
        dirnames[:] = filtered_dirnames
        if Path(dirpath) != root:
            dir_count += 1
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = _relative_path(root, path)
            try:
                mode = os.lstat(path).st_mode
            except OSError as exc:
                skipped_rows.append({"relative_path": rel, "reason": type(exc).__name__})
                continue
            if stat.S_ISLNK(mode):
                skipped_rows.append({"relative_path": rel, "reason": "symlink"})
                continue
            if not stat.S_ISREG(mode):
                skipped_rows.append({"relative_path": rel, "reason": "non_regular"})
                continue
            try:
                size = os.stat(path).st_size
            except OSError as exc:
                skipped_rows.append({"relative_path": rel, "reason": "stat_error"})
                error_rows.append(
                    {
                        "relative_path": rel,
                        "reason": "stat_error",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                skipped_rows.append(
                    {"relative_path": rel, "reason": "sha256_read_error"}
                )
                error_rows.append(
                    {
                        "relative_path": rel,
                        "reason": "sha256_read_error",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            total_bytes += size
            file_rows.append(
                {
                    "relative_path": rel,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
            try:
                scan_counts = _scan_file(
                    path,
                    {
                        "secret": SECRET_PATTERNS,
                        "local_path": LOCAL_PATH_PATTERNS,
                    },
                )
            except OSError as exc:
                error_rows.append(
                    {
                        "relative_path": rel,
                        "reason": "scan_read_error",
                        "error_type": type(exc).__name__,
                    }
                )
                secret_scan.add_error(rel)
                local_path_scan.add_error(rel)
            else:
                secret_scan.add_file_result(rel, scan_counts["secret"])
                local_path_scan.add_file_result(rel, scan_counts["local_path"])

    index_lines = [
        f"{row['sha256']}  {row['size_bytes']}  {row['relative_path']}\n"
        for row in sorted(file_rows, key=lambda item: item["relative_path"])
    ]
    payload: dict[str, Any] = {
        "status": "complete_with_errors" if error_rows else "complete",
        "label": label,
        "root_name": root.name,
        "file_count": len(file_rows),
        "directory_count": dir_count,
        "total_bytes": total_bytes,
        "index_format": "sha256  size_bytes  relative_path",
        "index_sha256": _sha256_text("".join(index_lines)),
        "secret_scan": secret_scan.to_dict(),
        "local_path_scan": local_path_scan.to_dict(),
        "skipped": sorted(skipped_rows, key=lambda item: item["relative_path"]),
        "errors": sorted(error_rows, key=lambda item: item["relative_path"]),
    }
    if include_file_index:
        payload["files"] = sorted(file_rows, key=lambda item: item["relative_path"])
    return payload


class ScanSummary:
    def __init__(self) -> None:
        self.match_count = 0
        self.scanned_file_count = 0
        self.error_count = 0
        self.files: list[str] = []
        self.error_files: list[str] = []

    def add_file_result(self, relative_path: str, match_count: int) -> None:
        self.scanned_file_count += 1
        if match_count:
            self.match_count += match_count
            self.files.append(relative_path)

    def add_error(self, relative_path: str) -> None:
        self.error_count += 1
        self.error_files.append(relative_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_count": self.match_count,
            "matched_file_count": len(self.files),
            "matched_files": sorted(self.files),
            "scanned_file_count": self.scanned_file_count,
            "scan_error_count": self.error_count,
            "scan_error_files": sorted(self.error_files),
            "scan_status": (
                "complete_with_errors" if self.error_count else "complete"
            ),
            "scan_mode": "streaming",
            "snippets_included": False,
        }


def analyze_b_vs_d_pair(
    *,
    label: str,
    control_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    control = _load_json(control_path)
    candidate = _load_json(candidate_path)
    control_groups = _groups_by_target(control)
    candidate_groups = _groups_by_target(candidate)
    targets = sorted(set(control_groups) | set(candidate_groups))
    return {
        "label": label,
        "control_file_name": control_path.name,
        "candidate_file_name": candidate_path.name,
        "control_label": control.get("label"),
        "candidate_label": candidate.get("label"),
        "control_profile": control.get("profile"),
        "candidate_profile": candidate.get("profile"),
        "classification": "cross_mode_diagnostic",
        "raw_generation_parity_claim": False,
        "target_diagnostics": [
            _target_diagnostics(
                target=target,
                control_group=control_groups.get(target),
                candidate_group=candidate_groups.get(target),
            )
            for target in targets
        ],
    }


def _target_diagnostics(
    *,
    target: int,
    control_group: dict[str, Any] | None,
    candidate_group: dict[str, Any] | None,
) -> dict[str, Any]:
    if control_group is None or candidate_group is None:
        return {
            "output_token_target": target,
            "status": "missing_group",
            "control_group_present": control_group is not None,
            "candidate_group_present": candidate_group is not None,
        }

    control_source = _group_token_source(control_group)
    candidate_source = _group_token_source(candidate_group)
    combined_source = _combined_sequence_source(control_source, candidate_source)
    control_repeatability = _repeatability(control_group, control_source)
    candidate_repeatability = _repeatability(candidate_group, candidate_source)
    control_tokens = _first_token_sequence(control_group, control_source)
    candidate_tokens = _first_token_sequence(candidate_group, candidate_source)
    visible_similarity = _visible_answer_similarity(control_group, candidate_group)
    token_diagnostic: dict[str, Any]
    if combined_source in {"mixed_token_sources", "partially_unavailable"}:
        token_diagnostic = {
            "status": combined_source,
            "reason": "token sources are not comparable",
        }
    elif control_tokens is None or candidate_tokens is None:
        token_diagnostic = {"status": "missing_token_sequence"}
    else:
        token_diagnostic = _token_sequence_diagnostics(control_tokens, candidate_tokens)

    return {
        "output_token_target": target,
        "status": "analyzed",
        "raw_generation_parity_claim": False,
        "sequence_source": combined_source,
        "control_sequence_source": control_source,
        "candidate_sequence_source": candidate_source,
        "within_control_repeatability": control_repeatability,
        "within_candidate_repeatability": candidate_repeatability,
        "cross_mode_token_diagnostic": token_diagnostic,
        "normalized_visible_answer_similarity": visible_similarity,
        "parity_label": _parity_label(control_source, candidate_source),
    }


def _group_token_source(group: dict[str, Any]) -> str:
    request_body = group.get("request_body")
    return_token_ids = (
        isinstance(request_body, dict) and request_body.get("return_token_ids") is True
    )
    observations = _observations(group)
    if not observations:
        return "unavailable"
    observation_sources = {
        _observation_token_source(observation, return_token_ids=return_token_ids)
        for observation in observations
    }
    if observation_sources == {"raw_stream_token_ids"}:
        return "raw_stream_token_ids"
    if observation_sources == {"retokenized_visible_output"}:
        return "retokenized_visible_output"
    if "raw_stream_token_ids" in observation_sources:
        return "mixed_token_sources"
    if len(observation_sources - {"unavailable"}) > 1:
        return "mixed_token_sources"
    if "retokenized_visible_output" in observation_sources:
        return "mixed_token_sources"
    return "unavailable"


def _combined_sequence_source(control_source: str, candidate_source: str) -> str:
    if control_source == candidate_source:
        return control_source
    if "unavailable" in {control_source, candidate_source}:
        return "partially_unavailable"
    return "mixed_token_sources"


def _parity_label(control_source: str, candidate_source: str) -> str:
    source = _combined_sequence_source(control_source, candidate_source)
    if source == "raw_stream_token_ids":
        return "raw_token_ids_cross_mode_diagnostic"
    if source == "retokenized_visible_output":
        return "retokenized_visible_output_cross_mode_diagnostic"
    if source == "mixed_token_sources":
        return "mixed_token_sources_cross_mode_diagnostic"
    if source == "partially_unavailable":
        return "partially_unavailable_cross_mode_diagnostic"
    return "token_sequence_unavailable_cross_mode_diagnostic"


def _repeatability(group: dict[str, Any], source: str) -> dict[str, Any]:
    observations = _observations(group)
    if source not in {"raw_stream_token_ids", "retokenized_visible_output"}:
        return {
            "status": source,
            "reason": "token source is not repeatability-comparable",
            "observation_count": len(observations),
        }
    sequences = _token_sequences_for_source(group, source)
    if not sequences:
        return {
            "status": "missing",
            "observation_count": len(observations),
        }
    hashes = [_token_sequence_hash(sequence) for sequence in sequences]
    unique_hashes = sorted(set(hashes))
    return {
        "status": "passed" if len(unique_hashes) == 1 else "failed",
        "observation_count": len(sequences),
        "unique_sequence_count": len(unique_hashes),
        "unique_sequence_hashes": unique_hashes,
        "sequence_lengths": [len(sequence) for sequence in sequences],
    }


def _token_sequence_diagnostics(
    control_tokens: list[int],
    candidate_tokens: list[int],
) -> dict[str, Any]:
    max_len = max(len(control_tokens), len(candidate_tokens))
    lcp = 0
    for left, right in zip(control_tokens, candidate_tokens):
        if left != right:
            break
        lcp += 1
    same_position_matches = sum(
        1 for left, right in zip(control_tokens, candidate_tokens) if left == right
    )
    exact = control_tokens == candidate_tokens
    first_divergence = None if exact else lcp
    return {
        "exact_match": exact,
        "control_length": len(control_tokens),
        "candidate_length": len(candidate_tokens),
        "output_length_equal": len(control_tokens) == len(candidate_tokens),
        "longest_common_prefix_tokens": lcp,
        "first_divergence_position_0_based": first_divergence,
        "control_token_at_first_divergence": _token_at(control_tokens, first_divergence),
        "candidate_token_at_first_divergence": _token_at(candidate_tokens, first_divergence),
        "same_position_token_matches": same_position_matches,
        "matching_token_percentage_same_position_over_max_len": (
            same_position_matches / max_len if max_len else 1.0
        ),
    }


def _token_at(tokens: list[int], index: int | None) -> int | None:
    if index is None or index >= len(tokens):
        return None
    return tokens[index]


def _visible_answer_similarity(
    control_group: dict[str, Any],
    candidate_group: dict[str, Any],
) -> dict[str, Any]:
    control_text = _first_visible_content(control_group)
    candidate_text = _first_visible_content(candidate_group)
    control_sha = _first_observation_sha(control_group)
    candidate_sha = _first_observation_sha(candidate_group)
    if control_text is None or candidate_text is None:
        return {
            "status": "unavailable",
            "reason": "visible_content_not_preserved",
            "output_sha256_equal": (
                control_sha == candidate_sha if control_sha and candidate_sha else None
            ),
        }
    control_normalized = _normalize_visible_answer(control_text)
    candidate_normalized = _normalize_visible_answer(candidate_text)
    return {
        "status": "computed",
        "ratio": difflib.SequenceMatcher(
            None,
            control_normalized,
            candidate_normalized,
            autojunk=False,
        ).ratio(),
        "control_normalized_sha256": _sha256_text(control_normalized),
        "candidate_normalized_sha256": _sha256_text(candidate_normalized),
        "output_sha256_equal": (
            control_sha == candidate_sha if control_sha and candidate_sha else None
        ),
    }


def _normalize_visible_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _first_visible_content(group: dict[str, Any]) -> str | None:
    for observation in _observations(group):
        value = observation.get("visible_content")
        if isinstance(value, str) and value:
            return value
        result = observation.get("result")
        if isinstance(result, dict):
            result_value = result.get("visible_content")
            if isinstance(result_value, str) and result_value:
                return result_value
    return None


def _first_observation_sha(group: dict[str, Any]) -> str | None:
    observations = _observations(group)
    if not observations:
        return None
    value = observations[0].get("output_sha256")
    return value if isinstance(value, str) else None


def _first_token_sequence(group: dict[str, Any], source: str) -> list[int] | None:
    sequences = _token_sequences_for_source(group, source)
    return sequences[0] if sequences else None


def _token_sequences_for_source(group: dict[str, Any], source: str) -> list[list[int]]:
    if source not in {"raw_stream_token_ids", "retokenized_visible_output"}:
        return []
    sequences: list[list[int]] = []
    for observation in _observations(group):
        if source == "raw_stream_token_ids":
            sequence = _raw_token_sequence(observation)
        else:
            sequence = _output_token_sequence(observation)
        if sequence is not None:
            sequences.append(sequence)
    return sequences


def _observation_token_source(
    observation: dict[str, Any],
    *,
    return_token_ids: bool,
) -> str:
    if return_token_ids and _raw_token_sequence(observation) is not None:
        return "raw_stream_token_ids"
    if _output_token_sequence(observation) is not None:
        return "retokenized_visible_output"
    return "unavailable"


def _raw_token_sequence(observation: dict[str, Any]) -> list[int] | None:
    direct = _int_sequence_or_none(observation.get("raw_output_token_ids"))
    if direct is not None:
        return direct
    result = observation.get("result")
    if isinstance(result, dict):
        return _int_sequence_or_none(result.get("raw_output_token_ids"))
    return None


def _output_token_sequence(observation: dict[str, Any]) -> list[int] | None:
    return _int_sequence_or_none(observation.get("output_token_ids"))


def _int_sequence_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return list(value)


def _observations(group: dict[str, Any]) -> list[dict[str, Any]]:
    observations = group.get("observations")
    if not isinstance(observations, list):
        return []
    return [item for item in observations if isinstance(item, dict)]


def _groups_by_target(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return {}
    by_target: dict[int, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        target = group.get("output_token_target")
        if isinstance(target, int) and not isinstance(target, bool):
            by_target[target] = group
    return by_target


def _token_sequence_hash(tokens: list[int]) -> str:
    return _sha256_text(json.dumps(tokens, separators=(",", ":")))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _reject_nonfinite_json_numbers(payload)
    return payload


def _reject_nonfinite_json_numbers(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_json_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json_numbers(item)


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _relative_error_path(root: Path, path: str | bytes | None) -> str:
    if path is None:
        return "."
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        return candidate.relative_to(root).as_posix()
    except (TypeError, ValueError):
        return "."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scan_file(
    path: Path,
    pattern_groups: dict[str, list[re.Pattern[bytes]]],
) -> dict[str, int]:
    counts = {name: 0 for name in pattern_groups}
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(SCAN_CHUNK_BYTES)
            if not chunk:
                break
            data = tail + chunk
            fresh_offset = len(tail)
            for name, patterns in pattern_groups.items():
                for pattern in patterns:
                    counts[name] += sum(
                        1
                        for match in pattern.finditer(data)
                        if match.end() > fresh_offset
                    )
            tail = data[-SCAN_OVERLAP_BYTES:]
    return counts


if __name__ == "__main__":
    main()
