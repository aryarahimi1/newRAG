#!/usr/bin/env bash
# Run the FastAPI backend (Svelte UI: `cd frontend && npm run dev` with proxy to :8000).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    echo "Creating .venv with python3.11…"
    python3.11 -m venv .venv
  else
    echo "Creating .venv with python3…"
    python3 -m venv .venv
  fi
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -q --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -q "posthog>=2.5.0,<4.0.0"

python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-cpu tensorflow-intel keras 2>/dev/null || true

exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"
