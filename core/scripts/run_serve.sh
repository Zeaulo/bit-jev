#!/usr/bin/env bash
# Start the TypeSafe-compatible server against a trained run.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/bit-jev-2b}
PORT=${2:-8009}
python -m bit_jev.serve --run "$RUN" --port "$PORT"
