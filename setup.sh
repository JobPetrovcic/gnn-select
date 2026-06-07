#!/usr/bin/env bash
# Self-contained setup for the GNN premise-selection service.
# Creates a local Python venv (CPU-only torch), installs deps, runs a self-test.
# Usage:  bash setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo "==> Python: $($PY --version)  (need 3.10–3.12)"

echo "==> Creating venv at .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null

echo "==> Installing CPU torch 2.4.1 (no CUDA needed)"
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
echo "==> Installing torch_geometric 2.7.0"
pip install torch_geometric==2.7.0

echo "==> Fetching model weights (2 x ~1.4 GB) if missing"
ART="$HERE/service/artifacts"
BASE="${GNN_ARTIFACTS_URL:-https://github.com/JobPetrovcic/gnn-select/releases/download/weights-v1}"
for f in premise_emb.pt rand_init_ctx.pt; do
  if [ -f "$ART/$f" ]; then
    echo "    have $f"
  else
    echo "    downloading $f ..."
    curl -L --fail -o "$ART/$f" "$BASE/$f"
  fi
done

echo "==> Running self-test"
bash "$HERE/selftest.sh"

echo
echo "Setup complete."
echo "  Start the service:  source $HERE/.venv/bin/activate && python $HERE/service/serve.py"
echo "  Lean side (heavy):  bash $HERE/setup_lean.sh    # installs Lean + mathlib snapshot"
