#!/usr/bin/env bash
# 2-minute end-to-end check: tokenizer, delimiter ids, block-causal mask, one forward.
set -euo pipefail
cd "$(dirname "$0")/.."
export BIT_JEV_ON_AUTODL=${BIT_JEV_ON_AUTODL:-1}
python -m bit_jev.train --data data/smoke.jsonl \
  --base microsoft/bitnet-b1.58-2B-4T-bf16 \
  --epochs 1 --lr 5e-5 --lora 16 --batch 2 --accum 1 --dtype bf16 --checkpointing 1 \
  --device cuda --out runs/smoke --log_every 1
python scripts/oracle_bitnet.py --run runs/smoke
