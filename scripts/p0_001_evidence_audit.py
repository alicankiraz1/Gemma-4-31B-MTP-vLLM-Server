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
    re.compile(rb"/Users/[A-Za-z0-9._/@%+=:,~-]+(?:/[A-Za-z0-9._/@%+=:,~-]+)*"),
    re.compile(rb"/home/[A-Za-z0-9._/@%+=:,~-]+(?:/[A-Za-z0-9._/@%+=:,~-]+)*"),
    re.compile(rb"/var/folders/[A-Za-z0-9._/@%+=:,~-]+(?:/[A-Za-z0-9._/@%+=:,~-]+)*"),
    re.compile(rb"file:///[A-Za-z0-9._/@%+=:,~-]+(?:/[A-Za-z0-9._/@%+=:,~-]+)*"),
    re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n\t \"']+"),
]


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
    args = parser.parse_args()

    payload = build_audit_payload(
        evidence_roots=[_parse_label_path(item) for item in args.evidence_root],
        pairs=[_parse_pair(item) for item in args.pair],
        include_file_index=not args.omit_file_index,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def build_audit_payload(
    *,
    evidence_roots: list[tuple[str, Path]],
    pairs: list[tuple[str, Path, Path]],
    include_file_index: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
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
    dir_count = 0
    total_bytes = 0
    secret_scan = ScanSummary()
    local_path_scan = ScanSummary()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not (Path(dirpath) / dirname).is_symlink()
        )
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
            size = path.stat().st_size
            digest = _sha256_file(path)
            total_bytes += size
            file_rows.append(
                {
                    "relative_path": rel,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
            data = _read_bytes_for_scan(path)
            secret_scan.add_file(rel, data, SECRET_PATTERNS)
            local_path_scan.add_file(rel, data, LOCAL_PATH_PATTERNS)

    index_lines = [
        f"{row['sha256']}  {row['size_bytes']}  {row['relative_path']}\n"
        for row in sorted(file_rows, key=lambda item: item["relative_path"])
    ]
    payload: dict[str, Any] = {
        "label": label,
        "root_name": root.name,
        "file_count": len(file_rows),
        "directory_count": dir_count,
        "total_bytes": total_bytes,
        "index_format": "sha256  size_bytes  relative_path",
        "index_sha256": _sha256_text("".join(index_lines)),
        "secret_scan": secret_scan.to_dict(),
        "local_path_scan": local_path_scan.to_dict(),
        "skipped": skipped_rows,
    }
    if include_file_index:
        payload["files"] = sorted(file_rows, key=lambda item: item["relative_path"])
    return payload


class ScanSummary:
    def __init__(self) -> None:
        self.match_count = 0
        self.files: list[str] = []

    def add_file(self, relative_path: str, data: bytes, patterns: list[re.Pattern[bytes]]) -> None:
        file_matches = 0
        for pattern in patterns:
            file_matches += len(pattern.findall(data))
        if file_matches:
            self.match_count += file_matches
            self.files.append(relative_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_count": self.match_count,
            "matched_file_count": len(self.files),
            "matched_files": sorted(self.files),
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
    control_repeatability = _repeatability(control_group)
    candidate_repeatability = _repeatability(candidate_group)
    control_tokens = _first_token_sequence(control_group)
    candidate_tokens = _first_token_sequence(candidate_group)
    visible_similarity = _visible_answer_similarity(control_group, candidate_group)
    token_diagnostic: dict[str, Any]
    if control_tokens is None or candidate_tokens is None:
        token_diagnostic = {"status": "missing_token_sequence"}
    else:
        token_diagnostic = _token_sequence_diagnostics(control_tokens, candidate_tokens)

    return {
        "output_token_target": target,
        "status": "analyzed",
        "sequence_source": _combined_sequence_source(control_source, candidate_source),
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
    raw_sequences = [
        _raw_token_sequence(observation)
        for observation in observations
        if _raw_token_sequence(observation) is not None
    ]
    output_sequences = [
        _output_token_sequence(observation)
        for observation in observations
        if _output_token_sequence(observation) is not None
    ]
    if return_token_ids and raw_sequences and len(raw_sequences) == len(observations):
        return "raw_stream_token_ids"
    if output_sequences:
        return "retokenized_visible_output"
    return "unavailable"


def _combined_sequence_source(control_source: str, candidate_source: str) -> str:
    if control_source == candidate_source:
        return control_source
    if "unavailable" in {control_source, candidate_source}:
        return "partially_unavailable"
    return "mixed"


def _parity_label(control_source: str, candidate_source: str) -> str:
    source = _combined_sequence_source(control_source, candidate_source)
    if source == "raw_stream_token_ids":
        return "raw_token_ids_cross_mode_diagnostic"
    if source == "retokenized_visible_output":
        return "retokenized_visible_output_cross_mode_diagnostic"
    return "token_sequence_unavailable_cross_mode_diagnostic"


def _repeatability(group: dict[str, Any]) -> dict[str, Any]:
    sequences = [
        sequence
        for sequence in (_token_sequence(observation) for observation in _observations(group))
        if sequence is not None
    ]
    if not sequences:
        return {
            "status": "missing",
            "observation_count": len(_observations(group)),
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


def _first_token_sequence(group: dict[str, Any]) -> list[int] | None:
    for observation in _observations(group):
        sequence = _token_sequence(observation)
        if sequence is not None:
            return sequence
    return None


def _token_sequence(observation: dict[str, Any]) -> list[int] | None:
    return _raw_token_sequence(observation) or _output_token_sequence(observation)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_bytes_for_scan(path: Path) -> bytes:
    return path.read_bytes()


if __name__ == "__main__":
    main()
