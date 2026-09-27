"""LoRA fine-tune of a BitNet b1.58 base into a bit-jev decision model (pointer head trained from scratch).

The release recipe (Path B -- what we publish and export to bitnet.cpp I2_S):

    python -m bit_jev.train \
        --suite data/decision-v7 --suite_split train \
        --base microsoft/bitnet-b1.58-2B-4T-bf16 \
        --epochs 2 --lr 5e-5 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 \
        --device cuda --out runs/bit-jev-2b

Fast checks:

    python -m bit_jev.train --data data/smoke.jsonl --epochs 1 --batch 2 --accum 1 \
        --device cuda --out runs/smoke            # minutes on one GPU

The backbone is loaded from the FP UNQUANTIZED bf16 checkpoint (`microsoft/bitnet-b1.58-2B-4T-bf16`,
whose weights are exactly ternary {-1, 0, +1}) and frozen; the accepted wisdom (BitNet paper, section
on partial fine-tuning) is that quantize->LoRA-tune preserves the CPU-path losslessness, and
`oracle_bitnet.py` asserts every frozen weight is still in {-1, 0, 1} after a run.

Batch size is small (variable-length records with custom masks) and gradients accumulate over
--accum micro-batches, as in kev.
"""
import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from . import DEFAULT_BASE
from . import hub as _hub
from .api import load_requests, read_jsonl, to_record, write_json
from .checkpoint import BitJevCheckpoint, Meta, write_meta
from .model import (MAX_STATE, DecisionModel, fits, load_tokenizer, delimiter_ids)


# --- small utils ----------------------------------------------------------------------

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def allocated_bytes(dev):
    if dev == "cuda":
        return torch.cuda.max_memory_allocated()
    if dev == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- data -------------------------------------------------------------------------------

def load_suite_requests(suite, split):
    """A frozen kev suite (v3 format: {"state", "instructions", "options" per row is kev-internal;
    v4+ are plain SystemOne-labelled JSONL). This loader reads the plain SystemOne form."""
    rows = read_jsonl(Path(suite) / f"{split}.jsonl")
    for r in rows:
        r.setdefault("_meta", {})
        r["_meta"].setdefault("id", f"{suite}/{split}/{r.get('_meta', {}).get('row', 'row')}")
        r["_meta"].setdefault("source", r.get("_meta", {}).get("source", Path(suite).name))
    return rows


# --- losses -----------------------------------------------------------------------------

def question_loss(z, q, dev, ord_w=0.0):
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


# --- batches ----------------------------------------------------------------------------

@dataclass(eq=False)
class Variant:
    req: dict
    rec: dict
    enc: dict

    @property
    def tokens(self):
        return len(self.enc["ids"])


def encode_chunk(model, tok, chunk, max_state):
    out = []
    for req in chunk:
        rec, _ = to_record(req, labelled=True)
        enc = model.encode(tok, rec, strict=True, max_state=max_state)
        out.append(Variant(req, rec, enc))
    return out


def batch_loss(model, batch, dev, ord_w, autocast):
    with autocast:
        logits_b = model.forward_batch([v.enc for v in batch])
    loss, terms, n = 0.0, Counter(), 0
    for v, logits in zip(batch, logits_b):
        ce = sum(question_loss(z.float(), q, dev, ord_w) for z, q in zip(logits, v.rec["questions"])) / len(logits)
        loss = loss + ce
        terms["ce"] += ce.item()
        n += 1
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    return loss, terms, n


# --- run -----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bit_jev.train")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="HF/ModelScope id or local dir of the FP bf16 BitNet checkpoint "
                         f"(default: {DEFAULT_BASE}; on modelscope: AI-ModelScope/bitnet-b1.58-2B-4T-bf16)")
    ap.add_argument("--base_revision", default="")
    ap.add_argument("--hub", choices=["hf", "modelscope", "auto"], default="auto",
                    help="download backend for base/tokenizer (auto picks modelscope on AutoDL, hf elsewhere)")
    ap.add_argument("--suite", default="", help="a kev-format frozen suite dir; trains on one of its splits")
    ap.add_argument("--suite_split", default="train")
    ap.add_argument("--data", default="", help="your own labelled JSONL (alternative to --suite)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=0.0, help="separate lr for the pointer head (0 = same as --lr)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4, help="records per forward pass")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--head_dim", type=int, default=256)
    ap.add_argument("--ord_w", type=float, default=0.0, help="ranked-probability-score weight for Score questions")
    ap.add_argument("--max_state", type=int, default=MAX_STATE)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16",
                    help="bf16 = autocast forward with fp32 master weights (CUDA)")
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="bf16",
                    help="dtype the frozen backbone is stored at; bf16 halves memory and matches the published base")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--init_from", default="", help="warm-start LoRA + head from an existing bit-jev run (delta fine-tune)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bit-jev")
    ap.add_argument("--log_every", type=int, default=10)
    a = ap.parse_args(argv)
    if bool(a.suite) == bool(a.data):
        ap.error("exactly one of --suite / --data is required")
    if a.epochs < 1 or a.accum < 1 or a.batch < 1 or a.lora < 1:
        ap.error("epochs/accum/batch/lora must be >= 1")
    if a.dtype == "bf16" and a.device == "cpu":
        ap.error("--dtype bf16 requires a GPU (cuda/mps)")
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0 or a.ord_w < 0:
        ap.error("invalid learning rate or loss weight")
    if Path(a.out).exists():
        ap.error(f"refusing to overwrite existing run dir {a.out}")
    return a


def check_delimiters(model, tok, log_prefix=""):
    """Verify the five delimiter ids resolve to distinct rows in the loaded model's embedding.

    Earlier drafts asserted the embeddings were exactly ternary ({-1,0,+1}). That was wrong: the
    microsoft/BitNet-b1.58-2B-4T-bf16 checkpoint stores bf16 weights, and the model's online
    dequantizer (transformers' bitnet integration) rounds them to ternary at every forward pass.
    The Path-B claim survives (the LLAMA weight matrices quantize back identically after training
    because LoRA never touches them); the embedding rows simply are the model's continuous
    representations of those tokens, exactly as the BitNet authors ship them.
    """
    emb = model.lm.get_input_embeddings().weight.detach()
    rows = emb[delimmap(tok)].float().cpu()
    distinct = len(torch.unique(rows, dim=0))
    if distinct < 5:
        raise ValueError(f"{log_prefix}delimiter rows are not distinct (only {distinct} of 5 unique); "
                         "the model cannot keep the branch boundaries apart.")
    print(f"{log_prefix}delimiter rows distinct: {distinct}/5 ok (id {delimmap(tok)})", flush=True)


def delimmap(tok):
    return delimiter_ids(tok)


def main(argv=None):
    a = parse_args(argv)
    if a.hub != "auto":
        os.environ["BIT_JEV_HUB"] = a.hub   # hub.py reads this once per process
    out_dir = Path(a.out); out_dir.mkdir(parents=True)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device or default_device()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.dtype == "bf16" and dev == "cuda") else contextlib.nullcontext()
    revision = a.base_revision or None

    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim,
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    check_delimiters(model, tok)

    meta = Meta(base=a.base, base_revision=revision, lora=a.lora, head_dim=a.head_dim)
    init_source = None
    if a.init_from:
        init_source = BitJevCheckpoint(a.init_from).warm_start(model, meta)
        print(f"delta: warm start from {init_source['resolved']} ({init_source['adapter_tensors']} adapter tensors)", flush=True)
    n_train = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"device={dev} base={a.base} trainable={n_train:.1f}M", flush=True)

    reqs = load_suite_requests(a.suite, a.suite_split) if a.suite else load_requests(a.data)
    rng.shuffle(reqs)
    kept = []
    for r in reqs:
        rec, _ = to_record(r, labelled=True)
        if fits(rec, tok, max_state=a.max_state):
            kept.append(r)
    if len(kept) < len(reqs):
        print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records outside the training context", flush=True)
    reqs = kept
    if not reqs:
        raise ValueError("empty training set after the context filter")
    qtypes = Counter(q["type"] for r in reqs for q in r["questions"].values())
    print(f"{len(reqs)} training records, questions by type {dict(qtypes)}", flush=True)

    suite_manifest = Path(a.suite) / "manifest.json" if a.suite else None
    write_json(out_dir / "training_config.json",
               {"args": vars(a), "base_revision": revision, "init_source": init_source,
                "data_digest": digest(suite_manifest) if (suite_manifest and suite_manifest.exists()) else None,
                "data_digest_data": digest(a.data) if a.data else None})

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    micro_per_epoch = math.ceil(len(reqs) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr],
                                                total_steps=max(steps, 1), pct_start=0.1)
    log_f = open(out_dir / "train.log", "w", encoding="utf-8")

    model.train()
    t0 = time.time(); run = Counter(); step = seen = tokens_seen = peak_mem = 0
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        for mb in range(micro_per_epoch):
            chunk = reqs[mb * a.batch: (mb + 1) * a.batch]
            batch = encode_chunk(model, tok, chunk, a.max_state)
            loss, terms, n = batch_loss(model, batch, dev, a.ord_w, autocast)
            (loss / a.accum).backward()
            run += terms; run["n"] += n; seen += n; tokens_seen += sum(v.tokens for v in batch)
            peak_mem = max(peak_mem, allocated_bytes(dev))
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % a.log_every == 0:
                    msg = (f"ep{ep} step {step}/{steps} loss {run['ce']/max(run['n'],1):.4f} "
                           f"{(time.time()-t0)/seen:.3f}s/rec peak {peak_mem/1e9:.1f}GB")
                    print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
                    run = Counter()
    log_f.close()

    model.lm.save_pretrained(a.out)
    meta.head = model.head.state_dict()
    meta.extra = {"args": vars(a), "init_source": init_source}
    write_meta(a.out, meta)
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json",
               {"wall_seconds": time.time() - t0, "records_seen": seen, "optimizer_steps": step,
                "forward_tokens": tokens_seen, "peak_device_bytes": peak_mem,
                "peak_loop_gb": peak_mem / 1e9, "device": dev, "dtype": a.dtype, "batch": a.batch})
    print(cmd_line_hint(a, revision))
    print("saved", a.out, flush=True)


def cmd_line_hint(a, revision):
    return (f"next: python scripts/oracle_bitnet.py --run {a.out} && "
            f"python -m bit_jev.eval --run {a.out} --data <dev.jsonl> --out {a.out}-dev")


if __name__ == "__main__":
    main()
