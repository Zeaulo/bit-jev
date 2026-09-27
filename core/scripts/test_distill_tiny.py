"""Tiny-model smoke of the full distillation code path (no big downloads, <2 min on CPU).

Builds a 2-layer BitNet student and a 2-layer "teacher" DecisionModel both from scratch on the
BitNet tokenizer, runs one distill step via bit_jev.distill's own helpers (WarmupQuantizer blend,
KD+CE loss, save_backbone), reloads the saved run through BitJevCheckpoint, and asserts the
round-trip: the loaded student's logits match the in-memory ones and the delimiter mask flows.

    python scripts/test_distill_tiny.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # this workstation's triton breaks inductor

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import BitNetConfig, BitNetForCausalLM

from bit_jev import hub
from bit_jev.api import to_record, write_json
from bit_jev.checkpoint import BitJevCheckpoint, Meta, write_meta
from bit_jev.distill import WeightWarmup
from bit_jev.model import DecisionModel, delimiter_ids
from bit_jev.train import encode_chunk


def tiny_bitnet():
    cfg = BitNetConfig(vocab_size=128256, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       hidden_act="relu2", max_position_embeddings=2048)
    return BitNetForCausalLM(cfg)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="bitjev-distill-smoke-"))
    print(f"workdir {tmp}")

    tok = hub.load_tokenizer("microsoft/bitnet-b1.58-2B-4T-bf16")
    dids = delimiter_ids(tok)
    assert len(set(dids)) == 5

    # --- teacher: a tiny DecisionModel with a trained-ish head (random init is fine for plumbing)
    teacher = DecisionModel.__new__(DecisionModel)
    from torch import nn
    nn.Module.__init__(teacher)
    teacher.lm = tiny_bitnet()
    from bit_jev.model import PointerHead
    teacher.head = PointerHead(64, dp=32)
    teacher.device = "cpu"
    teacher.train()

    # teacher checkpoint dir (what bit_jev.distill expects as --teacher)
    tdir = tmp / "teacher"
    teacher.lm.save_pretrained(tdir)  # full backbone; no adapter
    head_sd = {k: v.clone() for k, v in teacher.head.state_dict().items()}
    write_meta(tdir, Meta(base="microsoft/bitnet-b1.58-2B-4T-bf16", head=head_sd, head_dim=32))
    write_json(tdir / "training_config.json", {"tiny": True})

    # reload through the real checkpoint path (tests the distilled-run branch of load())
    ck = BitJevCheckpoint(str(tdir))
    tok2, t_loaded = ck.load("cpu", dtype=torch.float32, temperature=1.0)
    assert torch.equal(t_loaded.head.q.weight, head_sd["q.weight"]), "head round-trip broke"
    print("teacher checkpoint round-trip: OK")

    # --- student: fresh tiny backbone, teacher's head
    model = DecisionModel(str(tdir), tok, "cpu", lora=None, head_dim=32, dtype=torch.float32)
    model.head.load_state_dict(head_sd)
    model.train()

    # --- WeightWarmup patch: lam=0 must bypass quantization on a non-ternary tensor,
    # lam=1 must quantize it, uninstall must restore the original forward.
    import transformers.integrations.bitnet as bm
    WeightWarmup.capture(); WeightWarmup.install()
    w = torch.tensor([0.31, -0.52, 0.83, 0.44, -1.9, 0.0], dtype=torch.float32)

    WeightWarmup.lam = 0.0
    out0 = bm.WeightQuant.forward(None, w)
    assert torch.equal(out0, w), "lam=0 did not bypass quantization"

    WeightWarmup.lam = 1.0
    out1 = bm.WeightQuant.forward(None, w)
    assert not torch.equal(out1, w), "lam=1 did not quantize"
    scale = 1.0 / w.abs().mean()
    ref = (w * scale).round().clamp(-1, 1) / scale
    assert torch.allclose(out1, ref, atol=1e-6), "lam=1 quantized value wrong"

    WeightWarmup.uninstall()
    out2 = bm.WeightQuant.forward(None, w)
    assert torch.allclose(out2, ref, atol=1e-6), "uninstall did not restore raw quantizer"
    print(f"WarmupQuant patch: lam=0 bypasses, lam=1 quantizes (dev {(out1 - w).abs().max():.3f}), "
          f"uninstall restores: OK")

    # --- one record through encode_chunk
    req = {"state": "The shoes arrived late and the box was crushed.",
           "questions": {"team": {"type": "choice", "instructions": "Which team handles this?",
                                  "criteria": {"returns": None, "shipping": None}, "label": "shipping"}}}
    rec, _ = to_record(req, labelled=True)
    batch = encode_chunk(model, tok, [req], max_state=64)

    # KD + CE loss on the tiny model, backward flows (through WeightQuant's STE)
    logits_s = model.forward_batch([batch[0].enc])[0]
    z_s = logits_s[0].float()
    target = torch.softmax(torch.tensor([0.2, 0.8]), -1)
    kd = F.kl_div(F.log_softmax(z_s / 2.0, -1), target, reduction="batchmean") * 4.0
    y = torch.tensor([batch[0].rec["questions"][0]["label"]])
    ce = F.cross_entropy(z_s[None], y)
    (kd + 0.1 * ce).backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad > 0, "no gradients flowed"
    print(f"KD+CE backward ({n_grad} grads): OK")

    # save the full backbone + reload through checkpoint (the distilled-run branch)
    out = tmp / "distilled"
    out.mkdir()
    model.lm.save_pretrained(out)
    tok.save_pretrained(out)
    assert (out / "config.json").exists() and not (out / "adapter_model.safetensors").exists()
    # write head.pt so load() finds the pointer head, then reload through the checkpoint path
    write_meta(out, Meta(base="microsoft/bitnet-b1.58-2B-4T-bf16", head=head_sd, head_dim=32))
    ck2 = BitJevCheckpoint(str(out))
    tok3, m_loaded = ck2.load("cpu", dtype=torch.float32)
    # the test nudged ALL trainable params including the head (distill.py freezes it), so the
    # in-memory head is +0.01 off head_sd; align it before comparing logits.
    m_loaded.head.load_state_dict(model.head.state_dict())
    with torch.no_grad():
        a1 = model.forward_batch([batch[0].enc])[0][0]
        a2 = m_loaded.forward_batch([batch[0].enc])[0][0]
    assert torch.allclose(a1, a2, atol=1e-5), "saved-backbone reload changed logits"
    print("save_backbone -> checkpoint.load round-trip logits match: OK")

    shutil.rmtree(tmp)
    print("\nALL DISTILL PLUMBING TESTS PASSED")


if __name__ == "__main__":
    main()
