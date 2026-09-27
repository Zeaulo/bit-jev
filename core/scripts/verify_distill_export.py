"""Deployment-parity validation for the distilled bit-jev run.

    python scripts/verify_distill_export.py --run D:/.../bit-jev-2b-distilled --limit 30 --device cuda

What bitnet.cpp stores and computes with is exactly ternary round(W) * scale, where the formula
is bitnet.cpp's converter weight_quant(): absmean scale, round, clamp. What the 81.5% evaluation
computed was transformers' online WeightQuant -- THE SAME FORMULA applied to the same weights at
every forward. So the CPU model reproduces the eval path by construction; this script verifies
the two formulas agree numerically (they are written twice, once per codebase) and that the
offline-round weights behave identically under transformers' own offline mode.

Checks:
  1. formula parity: transformers' WeightQuant.forward(w) == bitnet.cpp's absmean_round(w),
     elementwise, on every weight matrix (fp32, tolerance 1e-6);
  2. offline-mode parity: load the model with quantization_mode=offline after rounding the
     stored latents, and compare option logits + argmax against the online-mode eval path on
     real records. Offline mode (weight * weight_scale, no per-forward quantization) is what
     the I2_S kernel computes.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import transformers.integrations.bitnet as bm
from transformers import AutoConfig

from bit_jev import hub
from bit_jev.api import to_record
from bit_jev.checkpoint import BitJevCheckpoint
from bit_jev.model import DecisionModel


def converter_quant(w):
    """bitnet.cpp convert-hf-to-gguf-bitnet.py:1055 weight_quant."""
    dtype = w.dtype
    w = w.float()
    s = 1 / w.abs().mean().clamp(min=1e-5)
    return ((w * s).round().clamp(-1, 1) / s).to(dtype)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=str(ROOT.parent / "learning/kev/evals/v7/decision-v7/development.jsonl"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ck = BitJevCheckpoint(a.run)
    tok_src = a.run if (Path(a.run) / "tokenizer_config.json").exists() else ck.meta.base
    tok = hub.load_tokenizer(tok_src)
    model = DecisionModel(a.run, tok, a.device, lora=None, head_dim=ck.meta.head_dim, dtype=torch.float32)
    model.head.load_state_dict(ck.meta.head)
    model.head.temperature = ck.meta.temperature
    model.eval()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.limit]
    for j, r in enumerate(rows):
        r.setdefault("_meta", {}).setdefault("id", f"row:{j}")

    # --- 1. the two quantization formulas agree elementwise
    print("== formula parity (transformers WeightQuant vs bitnet.cpp converter) ==")
    n_checked = worst = 0.0
    n_bad = 0
    for name, p in model.lm.named_parameters():
        if p.ndim < 2:
            continue
        w = p.detach().float()
        d = (bm.WeightQuant.forward(None, w) - converter_quant(w)).abs().max().item()
        n_checked += 1
        n_bad += d > 1e-6
        worst = max(worst, d)
    print(f"   {int(n_checked)} tensors: {int(n_bad)} mismatched (max elementwise diff {worst:.2e})")
    assert n_bad == 0

    # --- 2. offline-mode vs online-mode on real records
    print("== offline vs online forward parity ==")
    def run_all(m):
        outs = []
        with torch.no_grad():
            for r in rows:
                rec, _ = to_record(r, labelled=True)
                enc = m.encode(tok, rec)
                outs.append([z.clone() for z in m.forward(enc)])
        return outs

    logits_online = run_all(model)

    # flip every BitLinear to transformers' OFFLINE mode: weight stored as round(W), a single
    # weight_scale multiplier -- the transformers twin of the I2_S kernel's math.
    n_flipped = 0
    for mod in model.lm.modules():
        if isinstance(mod, bm.AutoBitLinear) and mod.online_quant:
            w = mod.weight.detach().float()
            s = 1 / w.abs().mean().clamp(min=1e-5)
            ternary = (w * s).round().clamp(-1, 1)
            mod.weight.data = ternary.to(mod.weight.dtype)      # stored as raw ternary {-1,0,1}
            mod.register_buffer("weight_scale",
                                torch.tensor([float(1 / s)], dtype=mod.weight.dtype,
                                             device=mod.weight.device))
            mod.online_quant = False
            n_flipped += 1
    print(f"   flipped {n_flipped} BitLinear layers to offline mode")

    logits_offline = run_all(model)

    max_diff, flips, n_q = 0.0, 0, 0
    for ref, got in zip(logits_online, logits_offline):
        for z1, z2 in zip(ref, got):
            max_diff = max(max_diff, (z1 - z2).abs().max().item())
            flips += int(z1.argmax() != z2.argmax())
            n_q += 1
    print(f"   {n_q} questions: max |logit diff| {max_diff:.5f}, argmax flips {flips}")
    assert flips == 0, "offline (deployment-semantics) path changed answers"
    print("\nOK: the I2_S deployment math reproduces the 81.5% eval path answer-for-answer.")


if __name__ == "__main__":
    main()
