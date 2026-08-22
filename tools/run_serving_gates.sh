#!/usr/bin/env bash
# Run tests/gate_serving against a live server, once per weight format.
#
# "Production ready" has to mean the same assertions pass on every runner, not
# on whichever one was developed last. This starts the server in each
# configuration -- AWQ and GGUF, both with the INT4 KV cache, the DFlash2
# drafter and the prefix cache live -- runs the same gate against each, and
# fails if any configuration fails.
#
# Timeouts are on everything. A server that does not come up in START_TIMEOUT,
# or a gate that does not finish in GATE_TIMEOUT, is a failure rather than a
# hang: this is meant to be safe to run unattended.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${QWEN38_WEIGHTS:-/home/patrickd/qwen38-weights}"
MODEL="${QWEN38_MODEL_DIR:-$W/Qwen3.8-27B-W4A16-AWQ}"
DRAFT="${QWEN38_DRAFT_DIR:-$W/Qwen3.8-27B-DFlash2}"
GGUF="${QWEN38_GGUF_Q3:-$W/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
PORT="${PORT:-8099}"
MAX_CTX="${MAX_CTX:-32768}"
LONG_WORDS="${LONG_WORDS:-8000}"
START_TIMEOUT="${START_TIMEOUT:-300}"
GATE_TIMEOUT="${GATE_TIMEOUT:-1800}"
LOGDIR="${LOGDIR:-$(mktemp -d)}"

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""; }
trap 'cleanup; exit 130' INT TERM

fail_total=0

run_config() {
  local name="$1"; shift
  local log="$LOGDIR/$name.server.log"
  echo
  echo "=============================================================="
  echo "  $name"
  echo "  server log: $log"
  echo "=============================================================="
  "$ROOT/build/cuda_server" --model "$MODEL" --port "$PORT" \
      --max-context "$MAX_CTX" --kv-cache int4 --embed-host \
      --draft "$DRAFT" "$@" > "$log" 2>&1 &
  SRV_PID=$!

  local waited=0
  until curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "  SERVER DIED DURING STARTUP"; tail -20 "$log"; fail_total=$((fail_total + 1))
      SRV_PID=""; return 1
    fi
    sleep 3; waited=$((waited + 3))
    if [ "$waited" -ge "$START_TIMEOUT" ]; then
      echo "  SERVER DID NOT COME UP IN ${START_TIMEOUT}s"; tail -20 "$log"
      cleanup; fail_total=$((fail_total + 1)); return 1
    fi
  done
  echo "  up in ${waited}s"

  timeout "$GATE_TIMEOUT" "$ROOT/build/gate_serving" --port "$PORT" --long-words "$LONG_WORDS"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  GATE FAILED (rc=$rc)"
    fail_total=$((fail_total + 1))
  fi
  # The server must still be alive: a gate that passes on a server that then
  # dies is not a pass.
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "  SERVER DIED DURING THE GATE"; tail -20 "$log"; fail_total=$((fail_total + 1)); SRV_PID=""
  fi
  cleanup
  sleep 3
  return 0
}

run_config "AWQ INT4 g128 + INT4 KV + DFlash2"
run_config "GGUF UD-Q3_K_XL + INT4 KV + DFlash2" --gguf "$GGUF"

echo
if [ "$fail_total" -eq 0 ]; then
  echo "ALL SERVING CONFIGURATIONS PASSED"
  exit 0
fi
echo "$fail_total CONFIGURATION(S) FAILED  (logs in $LOGDIR)"
exit 1
