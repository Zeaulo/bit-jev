"""End-to-end forward/backward smoke for the bit_jev training path on bitnet-b1.58-2B-4T-bf16.

    python scripts/test_workstation.py                # one minibatch (CPU: slow first time; ~5 GB download)
    python scripts/test_workstation.py --device cuda  # same on a GPU with at least 8 GB VRAM (fast)

Runs precisely what `bit_jev.train` does at init -- tokenizer, DecisionModel with rank-4 LoRA,
forward+backward on three smoke records -- so a regression in encode/mask/head/LoRA/delimiters
is caught before AutoDL time is spent on it. The heavyweight training goes through
`scripts/smoke_train.sh` on the GPU instance.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BIT_JEV_HUB", "modelscope" if os.environ.get("BIT_JEV_ON_AUTODL") else "hf")

from bit_jev import DEFAULT_BASE, hub
from bit_jev.api import load_requests, to_record
from bit_jev.model import DecisionModel, delimiter_ids


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lora", type=int, default=4)
    ap.add_argument("--bn", type=int, default=3, help="smoke records to run")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    print(f"== {DEFAULT_BASE}  (hub={hub.default_hub()}, device={a.device})")
    t0 = time.time()

    tok = hub.load_tokenizer(DEFAULT_BASE)
    dids = delimiter_ids(tok)
    print(f"   delimiters: {dids}")
    assert len(set(dids)) == 5 and all(128000 < i < 128256 for i in dids)

    print("   loading backbone + LoRA (this downloads ~5 GB on a cold cache)")
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = DecisionModel(DEFAULT_BASE, tok, a.device, lora=a.lora, dtype=dtype)
    model.train()

    reqs = load_requests(ROOT / "data" / "smoke.jsonl")[: a.bn]
    batch = []
    for r in reqs:
        rec, _ = to_record(r, labelled=True)
        batch.append((rec, model.encode(tok, rec)))

    logits_b = model.forward_batch([e for _, e in batch])
    loss, n = 0.0, 0
    for (rec, _), logits in zip(batch, logits_b):
        assert len(logits) == len(rec["questions"])
        for q, z in zip(rec["questions"], logits):
            y = torch.tensor([q["label"]], device=a.device)
            loss = loss + F.cross_entropy(z[None].float(), y)
            n += 1
    loss = loss / n
    print(f"   forward over {n} questions OK (loss {loss.item():.3f})")

    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"   backward populated {n_grad} trainable tensors in {time.time() - t0:.1f}s")
    print("OK: forward + backward both run end-to-end.")


if __name__ == "__main__":
    main()
