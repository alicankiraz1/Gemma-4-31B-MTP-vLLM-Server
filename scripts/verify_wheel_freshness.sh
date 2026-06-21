#!/usr/bin/env bash
set -euo pipefail

allow_dirty=0
if [[ "${1:-}" == "--allow-dirty" ]]; then
    allow_dirty=1
    shift
fi

if [[ "$allow_dirty" != "1" ]]; then
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "worktree is dirty; pass --allow-dirty for a non-release smoke" >&2
        exit 1
    fi
fi

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        python_bin=".venv/bin/python"
    elif [[ -x "../.venv/bin/python" ]]; then
        python_bin="../.venv/bin/python"
    else
        python_bin="python3"
    fi
fi

rm -f dist/gemma4_mtp_vllm-*.whl
"$python_bin" -m build --wheel

wheel=$(ls dist/gemma4_mtp_vllm-*.whl | head -n 1)
work=$(mktemp -d)
trap "rm -rf $work" EXIT

"$python_bin" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet "$wheel" fastapi httpx pytest pyyaml
cat <<'PY' > "$work/smoke.py"
import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def handler(request):
    if request.url.path in {"/health", "/v1/models", "/version"}:
        return httpx.Response(200, json={"status": "ok", "data": [], "version": "0.21.0"})
    return httpx.Response(404)


app = create_app(
    api_key="local-dev-key",
    vllm_base_url="http://vllm.local:8000",
    vllm_transport=httpx.MockTransport(handler),
)
client = TestClient(app)

livez = client.get("/livez")
assert livez.status_code == 200, livez.text

health = client.get("/health", headers={"x-api-key": "local-dev-key"})
assert health.status_code == 200, health.text
assert "Gemma 4 31B MTP" in health.text or "gemma-4-31B-it" in health.text

print("wheel smoke ok")
PY

"$work/venv/bin/python" "$work/smoke.py"
