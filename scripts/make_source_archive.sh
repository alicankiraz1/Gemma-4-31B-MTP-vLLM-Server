#!/usr/bin/env bash
set -euo pipefail

output="${1:-Gemma-4-31B-MTP-vllm-src.zip}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "worktree is dirty; refusing to create release archive" >&2
    exit 1
fi

git archive --format=zip --output "$output" HEAD
echo "wrote $output"
