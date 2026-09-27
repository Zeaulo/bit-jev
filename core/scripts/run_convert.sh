#!/usr/bin/env bash
# Path-B export to the bitnet.cpp layout (merged fp32 backbone + pointer head + pointer.json).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/bit-jev-2b}
OUT=${2:-export/bit-jev-2b}
python -m bit_jev.convert --run "$RUN" --out "$OUT"
python scripts/oracle_bitnet.py --run "$RUN" --export "$OUT"
echo "exported $RUN -> $OUT"
echo "feed $OUT/merged to bitnet.cpp:  python learning/bitnet/utils/convert-hf-to-gguf-bitnet.py $OUT/merged"
