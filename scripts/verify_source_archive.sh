#!/usr/bin/env bash
set -euo pipefail

archive="${1:?usage: verify_source_archive.sh <zip>}"

forbidden='(^|/)(\.git|\.venv|\.worktrees|\.pytest_cache|__pycache__|dist|build|__MACOSX|bench-results)(/|$)|\.pyc$|\.DS_Store$'

if unzip -Z1 "$archive" | grep -E "$forbidden" >/dev/null; then
    echo "archive contains forbidden entries"
    exit 1
fi
echo "archive clean: $archive"
