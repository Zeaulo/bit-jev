#!/usr/bin/env bash
# Benchmark against kev's frozen suites (in-distribution decision-v7 and out-of-distribution transfer-v4).
# Usage: bash scripts/run_eval.sh [run_dir]   (default runs/bit-jev-2b)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/bit-jev-2b}
export BIT_JEV_ON_AUTODL=${BIT_JEV_ON_AUTODL:-1}

[ -f data/decision-v7/development.jsonl ] || python scripts/fetch_kev_suites.py

echo "== in-distribution (decision-v7 development) =="
python -m bit_jev.eval --run "$RUN" --suite data/decision-v7 --suite_split development \
  --out "$RUN-dev-id"

echo "== OOD (transfer-v4 development) =="
python -m bit_jev.eval --run "$RUN" --suite data/transfer-v4 --suite_split development \
  --out "$RUN-dev-ood"
