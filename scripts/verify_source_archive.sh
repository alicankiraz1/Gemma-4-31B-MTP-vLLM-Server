#!/usr/bin/env bash
set -euo pipefail

archive="${1:?usage: verify_source_archive.sh <zip>}"

forbidden='(^|/)(\.git|\.venv|\.worktrees|\.pytest_cache|__pycache__|dist|build|__MACOSX|bench-results|artifacts|logs|docs/superpowers/plans|internal/superpowers)(/|$)|(^|/)(\.env[^/]*|[^/]*\.env[^/]*)(/|$)|\.pyc$|\.DS_Store$'

if [[ ! -f "$archive" ]]; then
    echo "archive not found: $archive" >&2
    exit 1
fi

unzip -t "$archive" >/dev/null

if unzip -Z1 "$archive" | grep -E "$forbidden" >/dev/null; then
    echo "archive contains forbidden entries"
    exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
unzip -qq "$archive" -d "$tmpdir"

local_user_path='/'"Users"'/'
local_home_path='/'"home"'/'
secret_like='(hf_[[:alnum:]_]{20,}|sk-proj-[[:alnum:]_-]{20,}|sk-[[:alnum:]_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY)'

if grep -R -I -E "(${local_user_path}|${local_home_path}|${secret_like})" "$tmpdir" >/dev/null; then
    echo "archive contains local or secret content"
    exit 1
fi
echo "archive clean: $archive"
