#!/usr/bin/env bash
# Starts the service, queries it for a known goal, and checks the expected premise
# is returned. Exits non-zero on failure. Activates .venv if present.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

PORT="${GNN_SVC_PORT:-8077}"
export GNN_SVC_PORT="$PORT"
python service/serve.py >/tmp/gnn_selftest.log 2>&1 &
SV=$!
trap 'kill $SV 2>/dev/null || true' EXIT

echo "==> Waiting for service on :$PORT (loads ~5s) ..."
for _ in $(seq 1 90); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

echo "health:   $(curl -s localhost:$PORT/health)"
RESP=$(curl -s "localhost:$PORT/retrieve" -H 'Content-Type: application/json' \
  -d '{"goal_constants":["Function.Injective","Finsupp","Finsupp.embDomain"],"lctx_constants":["Zero","Function.Embedding"],"k":5}')
echo "retrieve: $RESP"

if echo "$RESP" | grep -q '"Finsupp.embDomain_apply"'; then
  echo "==> SELF-TEST PASS (expected premise 'Finsupp.embDomain_apply' returned)"
else
  echo "==> SELF-TEST FAIL"; echo "service log:"; cat /tmp/gnn_selftest.log; exit 1
fi
