#!/usr/bin/env bash
set -euo pipefail

output="${1:-Gemma-4-31B-MTP-vllm-src.zip}"

git archive --format=zip --output "$output" HEAD
echo "wrote $output"
