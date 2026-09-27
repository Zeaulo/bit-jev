#!/usr/bin/env bash
# BitNet Path-B recipe: 2 epochs over the decision-v7 training partition, Kev's exact settings
# (lr 5e-5, LoRA r16, batch 4, accum 2, bf16 autocast, gradient checkpointing, OneCycle).
set -euo pipefail
cd "$(dirname "$0")/.."

export BIT_JEV_ON_AUTODL=${BIT_JEV_ON_AUTODL:-1}
[ -f data/smoke.jsonl ] || { echo "data/smoke.jsonl missing"; exit 1; }
python -m pip show modelscope >/dev/null || bash scripts/setup_autodl.sh

echo "== smoke =="
[ -d runs/smoke ] && echo "runs/smoke already trained; skipping smoke_train" || {
python -m bit_jev.train --data data/smoke.jsonl \
  --base microsoft/bitnet-b1.58-2B-4T-bf16 \
  --epochs 1 --lr 5e-5 --lora 16 --batch 2 --accum 1 --dtype bf16 --checkpointing 1 \
  --device cuda --out runs/smoke --log_every 1
}

echo "== release recipe (Kev's exact settings, ~1-2 h on a 24 GB card) =="
[ -f data/decision-v7/train.jsonl ] || python scripts/fetch_kev_suites.py
[ -d runs/bit-jev-2b ] && { echo "runs/bit-jev-2b already exists; refusing to overwrite"; exit 1; } || true
python -m bit_jev.train --suite data/decision-v7 --suite_split train \
  --base microsoft/bitnet-b1.58-2B-4T-bf16 \
  --epochs 2 --lr 5e-5 --lora 16 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 \
  --device cuda --out runs/bit-jev-2b

echo "== invariant check =="
python scripts/oracle_bitnet.py --run runs/bit-jev-2b
echo "== fit calibration temperature on dev =="
[ -d runs/bit-jev-2b-dev ] && echo "runs/bit-jev-2b-dev already trained; skipping" || {
python -m bit_jev.eval --run runs/bit-jev-2b \
  --suite data/decision-v7 --suite_split development \
  --fit_temperature --write_back --out runs/bit-jev-2b-dev
}
echo "trained. pull runs/bit-jev-2b back to the workstation for fp32 export."
