"""Distill a trained LoRA bit-jev run into the full BitNet backbone (Path B stage 2).

Two phases (they fit a 32 GB card; a single process holding teacher + student does not):

  # phase 1: teacher forward over the suite, teacher logits saved to disk (~2 MB)
  python -m bit_jev.distill export_teacher \
      --teacher runs/bit-jev-2b --suite data/decision-v7 --suite_split train \
      --device cuda --teacher_dtype bf16 --out runs/distill/teacher_logits.jsonl

  # phase 2: full-parameter QAT of the backbone against the saved logits
  python -m bit_jev.distill train \
      --teacher_logits runs/distill/teacher_logits.jsonl \
      --suite data/decision-v7 --suite_split train \
      --head_from runs/bit-jev-2b \
      --epochs 2 --lr 2e-5 --batch 2 --accum 4 --device cuda \
      --out runs/bit-jev-2b-distilled

Why: the LoRA delta cannot be merged into a BitNet base -- the online ternary quantizer rounds
it away at every forward (73.1% unmerged vs 32.4% merged on decision-v7 dev). To get the
decision skill INTO the ternary weights we fine-tune the backbone itself (full-parameter QAT,
the HF1BitLLM recipe): gradients flow through the online quantizer's STE into the continuous
latents, the pointer head stays frozen (deployment uses the SAME head), and the target is the
teacher's answer distribution (KD on the option logits, plus a small hard-label CE).

The starting latents are already quantization-ready -- quant(checkpoint) IS the deployed
behavior -- so warmup defaults to 0 (pure QAT from step 0). --warmup_steps > 0 blends the
quantizer in linearly, HF1BitLLM-style, by patching WeightQuant:
    forward := w + lam * (quant(w) - w),  lam: 0 -> 1
which needs no copy of the base weights.

Phase-2 memory (32 GB card): student fp32 9.7 + grads 9.7 + 8-bit AdamW ~4.8 + activations
2-3 = ~27 GB with batch 2. Requires bitsandbytes for the 8-bit optimizer (fp32 AdamW would
need ~39 GB). The saved run is a FULL backbone + head.pt: checkpoints.load detects it, and
round(W) -> {-1,0,+1} equals the deployment forward, so the I2_S conversion is lossless.
"""
import argparse
import contextlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from . import hub as _hub
from .api import load_requests, read_jsonl, to_record, write_json
from .checkpoint import BitJevCheckpoint, Meta, write_meta
from .model import DecisionModel, fits, load_tokenizer
from .train import allocated_bytes, default_device, digest, encode_chunk


# --- quantizer warmup (optional) ---------------------------------------------------------

class WeightWarmup:
    """Patch transformers' WeightQuant.forward with the HF1BitLLM blend:
    w + lam * (quant(w) - w). lam lives on the class; install() before training,
    set lam per step, uninstall() after. lam >= 1 short-circuits to the original."""

    lam = 1.0
    _installed = False

    @classmethod
    def install(cls):
        if cls._installed:
            return
        import transformers.integrations.bitnet as bitnet_mod
        orig = bitnet_mod.WeightQuant.forward

        def patched(ctx, weight):
            w_q = orig(ctx, weight)
            lam = cls.lam
            if lam >= 1.0:
                return w_q
            return weight + lam * (w_q - weight)

        bitnet_mod.WeightQuant.forward = staticmethod(patched)
        cls._installed = True

    @classmethod
    def uninstall(cls):
        if not cls._installed:
            return
        import transformers.integrations.bitnet as bitnet_mod
        # the original forward is restored by re-importing through the saved attribute
        bitnet_mod.WeightQuant.forward = cls._orig_forward
        cls._installed = False

    @classmethod
    def capture(cls):
        import transformers.integrations.bitnet as bitnet_mod
        cls._orig_forward = bitnet_mod.WeightQuant.forward


# --- shared data helpers -----------------------------------------------------------------

def load_suite_records(suite, suite_split, data, tok, max_state, limit, rng):
    """Shuffled, context-filtered records; each row carries a stable id for KD matching."""
    if bool(suite) == bool(data):
        raise ValueError("exactly one of --suite / --data is required")
    reqs = read_jsonl(Path(suite) / f"{suite_split}.jsonl") if suite else load_requests(data)
    for j, r in enumerate(reqs):
        r.setdefault("_meta", {}).setdefault("id", f"{suite or data}:{j}")
    rng.shuffle(reqs)
    kept = []
    for r in reqs:
        rec, _ = to_record(r, labelled=True)
        if fits(rec, tok, max_state=max_state):
            kept.append(r)
    if len(kept) < len(reqs):
        print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records outside the context", flush=True)
    if limit:
        kept = kept[:limit]
    if not kept:
        raise ValueError("empty record set after the context filter")
    return kept


def encode_batch(model, tok, chunk, max_state):
    out = []
    for req in chunk:
        rec, _ = to_record(req, labelled=True)
        enc = model.encode(tok, rec, strict=True, max_state=max_state)
        out.append({"req": req, "rec": rec, "enc": enc})
    return out


# --- phase 1: export teacher logits ------------------------------------------------------

def run_export_teacher(a):
    dev = a.device or default_device()
    ck = BitJevCheckpoint(a.teacher)
    tok, teacher = ck.load(dev, dtype=torch.bfloat16 if a.teacher_dtype == "bf16" else torch.float32,
                           temperature=1.0)   # raw logits: KD divides by T itself
    teacher.eval()
    print(f"teacher: {a.teacher} (base {ck.meta.base}, T forced to 1.0)", flush=True)

    rng = random.Random(a.seed)
    reqs = load_suite_records(a.suite, a.suite_split, a.data, tok, a.max_state, a.limit, rng)
    print(f"exporting logits for {len(reqs)} records", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_q = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f, torch.no_grad():
        for i, req in enumerate(reqs):
            rec, _ = to_record(req, labelled=True)
            enc = teacher.encode(tok, rec, strict=True, max_state=a.max_state)
            logits = teacher.forward(enc)
            row = {"record_id": req["_meta"]["id"], "questions": []}
            for q, z in zip(rec["questions"], logits):
                row["questions"].append({"qid": q["qid"] if "qid" in q else None,
                                         "label": q["label"], "qtype": q["qtype"],
                                         "logits": [round(float(x), 5) for x in z.float().cpu().tolist()]})
                n_q += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(reqs)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {n_q} teacher questions -> {out} ({time.time() - t0:.0f}s)", flush=True)


# --- phase 2: full-parameter QAT against saved logits -------------------------------------

def run_train(a):
    if a.warmup_steps:
        WeightWarmup.capture()
        WeightWarmup.install()
    dev = a.device or default_device()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.dtype == "bf16" and dev == "cuda") else contextlib.nullcontext()

    # KD targets, keyed by record id
    targets = {}
    for line in open(a.teacher_logits, encoding="utf-8"):
        row = json.loads(line)
        targets[row["record_id"]] = row["questions"]
    print(f"teacher logits: {len(targets)} records from {a.teacher_logits}", flush=True)

    head_ck = BitJevCheckpoint(a.head_from)
    base_name = a.base or head_ck.meta.base
    tok = load_tokenizer(base_name)

    rng = random.Random(a.seed)
    reqs = load_suite_records(a.suite, a.suite_split, a.data, tok, a.max_state, a.limit, rng)
    # keep only records that have KD targets
    reqs = [r for r in reqs if r["_meta"]["id"] in targets]
    if not reqs:
        raise ValueError("no records with teacher logits -- did phase 1 use the same suite/split?")
    print(f"{len(reqs)} QAT records", flush=True)

    out_dir = Path(a.out); out_dir.mkdir(parents=True)
    model = DecisionModel(base_name, tok, dev, lora=None, head_dim=head_ck.meta.head_dim,
                          dtype=torch.float32)
    model.head.load_state_dict(head_ck.meta.head)
    model.head.temperature = head_ck.meta.temperature
    if not a.train_head:
        for p in model.head.parameters():
            p.requires_grad_(False)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    head_state = "head+backbone" if a.train_head else "backbone, head frozen"
    print(f"student: full backbone trainable ({n_train:.2f}B fp32 latents; {head_state})", flush=True)

    if a.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise SystemExit("--optimizer adamw8bit needs bitsandbytes (pip install bitsandbytes); "
                             "fp32 AdamW does not fit a 32 GB card for a 2.4B backbone") from e
        opt = bnb.optim.AdamW8bit(model.trainable_parameters(), lr=a.lr, weight_decay=a.weight_decay)
        print("optimizer: AdamW8bit", flush=True)
    else:
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=a.weight_decay)
        print("optimizer: AdamW (fp32) -- make sure the card fits ~39 GB", flush=True)

    micro_per_epoch = math.ceil(len(reqs) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)

    write_json(out_dir / "distill_config.json",
               {"args": vars(a), "base": base_name, "head_from": a.head_from, "steps": steps,
                "teacher_logits": str(a.teacher_logits),
                "data_digest": digest(Path(a.suite) / "manifest.json") if a.suite else digest(a.data)})

    t0 = time.time(); run = Counter(); step = seen = tokens_seen = peak = 0
    log_f = open(out_dir / "distill.log", "w", encoding="utf-8")
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        for mb in range(micro_per_epoch):
            chunk = reqs[mb * a.batch: (mb + 1) * a.batch]
            batch = encode_batch(model, tok, chunk, a.max_state)
            WeightWarmup.lam = 1.0 if a.warmup_steps == 0 else min(step / a.warmup_steps, 1.0)
            with autocast:
                logits_s = model.forward_batch([b["enc"] for b in batch])
            loss = kd_sum = ce_sum = 0.0
            n_q = 0
            for b, ls in zip(batch, logits_s):
                tqs = targets[b["req"]["_meta"]["id"]]
                for z_s, tq in zip(ls, tqs):
                    z_s = z_s.float()
                    z_t = torch.tensor(tq["logits"], device=dev, dtype=torch.float32)
                    kl = F.kl_div(F.log_softmax(z_s / a.T, -1),
                                  F.softmax(z_t / a.T, -1), reduction="batchmean") * (a.T ** 2)
                    kd_sum = kd_sum + kl
                    y = torch.tensor([tq["label"]], device=dev)
                    ce_sum = ce_sum + F.cross_entropy(z_s[None], y)
                    n_q += 1
            kd = kd_sum / n_q; ce = ce_sum / n_q
            loss = a.kd_w * kd + a.alpha * ce
            (loss / a.accum).backward()
            run["kd"] += kd.item(); run["ce"] += ce.item(); run["n"] += n_q
            seen += len(batch); tokens_seen += sum(len(b["enc"]["ids"]) for b in batch)
            peak = max(peak, allocated_bytes(dev))
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % a.log_every == 0:
                    msg = (f"ep{ep} step {step}/{steps} lam={WeightWarmup.lam:.3f} "
                           f"kd {run['kd']/max(run['n'],1):.4f} ce {run['ce']/max(run['n'],1):.4f} "
                           f"{(time.time()-t0)/max(seen,1):.2f}s/rec peak {peak/1e9:.1f}GB")
                    print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
                    run = Counter()
    log_f.close()
    WeightWarmup.uninstall()

    # lam == 1 here: the latents are the final QAT weights; export rounds them losslessly.
    write_meta(out_dir, Meta(base=base_name, head=head_ck.meta.head, head_dim=head_ck.meta.head_dim,
                             temperature=head_ck.meta.temperature,
                             extra={"distilled_from": a.head_from,
                                    "teacher_logits_sha256": digest(a.teacher_logits),
                                    "args": vars(a)}))
    lm = model.lm
    lm.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    write_json(out_dir / "distill_metrics.json",
               {"wall_seconds": time.time() - t0, "optimizer_steps": step,
                "forward_tokens": tokens_seen, "peak_gb": peak / 1e9})
    print(f"saved distilled run -> {out_dir}", flush=True)
    print(f"next: python -m bit_jev.eval --run {out_dir} --suite {a.suite or '--data ' + a.data} "
          f"--out {out_dir}-dev", flush=True)


# --- CLI -----------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bit_jev.distill")
    sub = ap.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("export_teacher")
    pe.add_argument("--teacher", required=True)
    pe.add_argument("--suite", default=""); pe.add_argument("--suite_split", default="train")
    pe.add_argument("--data", default="")
    pe.add_argument("--teacher_dtype", choices=["fp32", "bf16"], default="bf16")
    pe.add_argument("--max_state", type=int, default=384)
    pe.add_argument("--limit", type=int, default=0)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    pe.add_argument("--out", default="runs/distill/teacher_logits.jsonl")

    pt = sub.add_parser("train")
    pt.add_argument("--teacher_logits", required=True)
    pt.add_argument("--head_from", required=True, help="LoRA run carrying the pointer head")
    pt.add_argument("--train_head", action="store_true",
                    help="keep the pointer head trainable during QAT (needed when the teacher's "
                         "hidden size differs, e.g. kev-9B -> BitNet KD; default freezes it)")
    pt.add_argument("--base", default="")
    pt.add_argument("--suite", default=""); pt.add_argument("--suite_split", default="train")
    pt.add_argument("--data", default="")
    pt.add_argument("--epochs", type=int, default=2)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--batch", type=int, default=2)
    pt.add_argument("--accum", type=int, default=4)
    pt.add_argument("--kd_w", type=float, default=1.0)
    pt.add_argument("--alpha", type=float, default=0.1)
    pt.add_argument("--T", type=float, default=2.0)
    pt.add_argument("--warmup_steps", type=int, default=0,
                    help="0 = pure QAT from the start (the checkpoint is already quantization-"
                         "ready); >0 anneals WeightQuant linearly, HF1BitLLM-style")
    pt.add_argument("--optimizer", choices=["adamw8bit", "adamw"], default="adamw8bit")
    pt.add_argument("--max_state", type=int, default=384)
    pt.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16",
                    help="autocast only; latents are always fp32")
    pt.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    pt.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    pt.add_argument("--limit", type=int, default=0)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--log_every", type=int, default=10)
    pt.add_argument("--out", default="runs/bit-jev-2b-distilled")

    a = ap.parse_args(argv)
    if a.mode == "export_teacher" and bool(a.suite) == bool(a.data):
        ap.error("exactly one of --suite / --data is required")
    if a.mode == "train" and not a.suite and not a.data:
        ap.error("train needs --suite or --data")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "export_teacher":
        run_export_teacher(a)
    else:
        run_train(a)


if __name__ == "__main__":
    main()
