#!/usr/bin/env bash
# Always run Streamlit with the project venv (not Homebrew Python).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating .venv with python3.11…"
  python3.11 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -q --upgrade pip
python -m pip install -r requirements.txt
# Chroma 0.5.x expects posthog.capture(distinct_id, event, props); posthog 4+ removed it.
python -m pip install -q "posthog>=2.5.0,<4.0.0"

# Stray TF installs break transformers; this project is torch-only.
python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-cpu tensorflow-intel keras 2>/dev/null || true

exec streamlit run streamlit_app.py "$@"
