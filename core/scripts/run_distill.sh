#!/usr/bin/env bash
# Route-3 distillation with the kev-9B teacher, on AutoDL (32 GB card, two isolated phases).
set -euo pipefail
cd "$(dirname "$0")/.."
export BIT_JEV_ON_AUTODL=${BIT_JEV_ON_AUTODL:-1}
KEV_VENV=${KEV_VENV:-/root/autodl-tmp/kev-venv}

echo "== phase 1: kev-9B teacher logits (kev venv, bf16, ~1 h) =="
[ -f runs/distill/kev9b_logits.jsonl ] || \
"$KEV_VENV/bin/python" scripts/export_kev_teacher.py \
  --teacher jaredpalmer/kev-9b \
  --suite data/decision-v7 --suite_split train \
  --out runs/distill/kev9b_logits.jsonl

echo "== phase 2: full-parameter QAT from kev logits (main env, ~2-4 h) =="
python -m pip show bitsandbytes >/dev/null 2>&1 || python -m pip install bitsandbytes
[ -d runs/bit-jev-2b-distilled ] || python -m bit_jev.distill train \
  --teacher_logits runs/distill/kev9b_logits.jsonl \
  --head_from runs/bit-jev-2b --train_head \
  --suite data/decision-v7 --suite_split train \
  --epochs 2 --lr 2e-5 --batch 2 --accum 4 --optimizer adamw8bit \
  --device cuda \
  --out runs/bit-jev-2b-distilled

echo "== evaluate the distilled model =="
[ -d runs/bit-jev-2b-distilled-dev ] || python -m bit_jev.eval --run runs/bit-jev-2b-distilled \
  --suite data/decision-v7 --suite_split development \
  --out runs/bit-jev-2b-distilled-dev
echo "distilled. pull runs/bit-jev-2b-distilled{,-dev} + runs/distill/kev9b_logits.jsonl back."
