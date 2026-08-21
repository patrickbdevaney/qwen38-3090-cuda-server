#!/usr/bin/env bash
# Generate GGML goldens: one representative tensor per quant type present in the
# file, dequantised by ggml itself. Writes <out>/<TYPE>.{f32,meta}.
set -euo pipefail
GGUF="${1:?usage: dump_gguf_ref.sh FILE.gguf OUT_DIR [max_elems]}"
OUT="${2:?}"
CAP="${3:-262144}"
BIN="$(dirname "$0")/../build"
mkdir -p "$OUT"
: > "$OUT/manifest.txt"
# Record the source file. The goldens are keyed by TENSOR NAME, and the same
# name holds different quantised data in every GGUF, so a gate run against a
# different file compares our dequantisation of one tensor against ggml's
# dequantisation of an unrelated one and reports a mismatch that means nothing.
# gate_gguf refuses to run unless this matches.
echo "# source $(basename "$GGUF")" >> "$OUT/manifest.txt"
# one tensor per type: first name seen for that type, preferring the largest
"$BIN/gguf_inspect" "$GGUF" --names | awk '/--- tensors ---/{f=1;next} f && NF>=3 {print $2, $1}' \
  | sort -u -k1,1 | while read -r TYPE NAME; do
  [ -z "$TYPE" ] && continue
  "$BIN/gguf_dequant_ref" "$GGUF" "$NAME" "$OUT/$TYPE.f32" "$CAP" >/dev/null
  N=$(stat -c%s "$OUT/$TYPE.f32"); N=$((N/4))
  echo "$TYPE $NAME $N" >> "$OUT/manifest.txt"
  printf '  %-10s %-40s %d elems\n' "$TYPE" "$NAME" "$N"
done
echo "wrote $OUT"
