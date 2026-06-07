#!/usr/bin/env bash
# Bootstrap the Lean side: install elan (if needed), create a lake project pinned to
# the corpus snapshot (mathlib 29dcec07 / lean v4.10.0-rc1), fetch the prebuilt cache,
# and drop in GnnSelect.lean. Heavy: downloads mathlib (GBs) and may build if no cache.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$HERE/lean/gnn_lean_project"
COMMIT="29dcec074de168ac2bf835a77ef68bbe069194c5"
TOOLCHAIN="leanprover/lean4:v4.10.0-rc1"

# 1. elan (Lean toolchain manager) — installs the toolchain on first lake call
if ! command -v lake >/dev/null 2>&1; then
  echo "==> Installing elan (Lean toolchain manager)"
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  # shellcheck disable=SC1090
  source "$HOME/.elan/env"
fi

# 2. lake project pinned to the corpus snapshot
mkdir -p "$PROJ"
cd "$PROJ"
echo "$TOOLCHAIN" > lean-toolchain
cat > lakefile.lean <<EOF
import Lake
open Lake DSL

package «gnn_lean_project»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "$COMMIT"

@[default_target]
lean_lib «GnnSelect»
EOF
cp "$HERE/lean/GnnSelect.lean" GnnSelect.lean

echo "==> Resolving mathlib @ $COMMIT (downloads source) ..."
lake update
echo "==> Downloading prebuilt mathlib cache (large; may 404 for old commits) ..."
if ! lake exe cache get; then
  echo "WARNING: 'lake exe cache get' failed for this commit."
  echo "         You can still 'lake build' but it will COMPILE mathlib (hours)."
fi
echo "==> Compiling GnnSelect (works even without the service running) ..."
lake build GnnSelect || echo "NOTE: build needs mathlib oleans; run 'lake build' after the cache/build completes."

echo
echo "Lean project ready: $PROJ"
echo "  • VSCode: open that folder with the Lean4 extension, open GnnSelect.lean"
echo "  • CLI:    cd $PROJ && lake env lean GnnSelect.lean   (with the Python service running)"
