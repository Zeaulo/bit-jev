"""Export teacher logits from an official kev checkpoint (jaredpalmer/kev-9b) for bit-jev distillation.

    # on AutoDL, in the kev venv (transformers 5.17+ + flash-linear-attention):
    python scripts/export_kev_teacher.py \
        --teacher jaredpalmer/kev-9b \
        --suite data/decision-v7 --suite_split train \
        --out runs/distill/kev9b_logits.jsonl

Runs the kev package's own scoring path (its tokenizer, its Qwen delimiters, its hybrid-backbone
row form) over the same records bit_jev trains on, and writes one JSONL row per record with the
per-question option logits keyed by (record_id, qid, qtype, label) -- exactly the schema
`bit_jev.distill train --teacher_logits` consumes. The two tokenizers never meet: KD aligns
option INDICES, both sides read the same state/options text.

Run this inside a separate venv from the BitNet training env (kev needs transformers>=5.17 and
flash-linear-attention; bit-jev pins 4.52-4.57 for BitNetForCausalLM).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="jaredpalmer/kev-9b")
    ap.add_argument("--suite", default="data/decision-v7")
    ap.add_argument("--suite_split", default="train")
    ap.add_argument("--data", default="", help="labelled JSONL alternative to --suite")
    ap.add_argument("--max_state", type=int, default=384, help="kev training context state cap")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/distill/kev9b_logits.jsonl")
    return ap.parse_args(argv)


def load_records(a):
    """The same records bit_jev trains on, in SystemOne request shape with labels."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bit_jev.api import read_jsonl, load_requests, to_record  # api has no heavy imports

    reqs = read_jsonl(Path(a.suite) / f"{a.suite_split}.jsonl") if a.suite else load_requests(a.data)
    for j, r in enumerate(reqs):
        r.setdefault("_meta", {}).setdefault("id", f"{a.suite or a.data}:{j}")
    # kev context filter: use kev's own fits() through its model if available; here the static
    # training-context rule (same numbers bit_jev uses) applied via kev.model.encode strict.
    rng = random.Random(a.seed)
    rng.shuffle(reqs)
    if a.limit:
        reqs = reqs[: a.limit]
    return reqs


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    # bit_jev.api lives next to this script's repo root; kev's heavy deps are not needed for it,
    # but it must be importable in the kev venv too, so resolve the path rather than assume CWD.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bit_jev.api import to_record

    # --- kev path (separate venv; heavy imports happen here, after our api import above)
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import MAX_STATE, MAX_BRANCH, MAX_PACKED, ContextOverflow

    ck = Checkpoint(a.teacher)
    # kev's default LoadOptions is fp32 (the exact-eval path, ~36 GB for kev-9B); bf16 merged is
    # the documented serving path (~18 GB) and what fits a 32 GB card. KD logits need consistency,
    # not fp32 exactness, and KD divides by its own T anyway.
    tok, model = ck.load(a.device, LoadOptions(dtype=torch.bfloat16))
    model.eval()
    print(f"teacher: {a.teacher} loaded on {a.device} (kev temperature {model.head.temperature})", flush=True)

    reqs = load_records(a)
    print(f"exporting logits for {len(reqs)} records", flush=True)

    n_q = dropped = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8", newline="\n") as f, torch.no_grad():
        for i, req in enumerate(reqs):
            rec, _ = to_record(req, labelled=True)
            # our internal record -> kev record shape: {"state", "questions": [{instr, options, label}]}
            kev_rec = {"state": rec["state"],
                       "questions": [{"instr": q["instr"], "options": q["options"],
                                      "label": q["label"], "qtype": q["qtype"]} for q in rec["questions"]]}
            try:
                enc = model.encode(tok, kev_rec, strict=True,
                                   max_state=MAX_STATE, max_branch=MAX_BRANCH)
                if len(enc["ids"]) > MAX_PACKED:
                    raise ContextOverflow("packed")
                logits = model.forward(enc)
            except (ContextOverflow, ValueError):
                dropped += 1
                continue
            row = {"record_id": req["_meta"]["id"], "questions": []}
            for q, z in zip(kev_rec["questions"], logits):
                row["questions"].append({"label": q["label"], "qtype": q["qtype"],
                                         "logits": [round(float(x), 5) for x in z.float().cpu().tolist()]})
                n_q += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(reqs)}  ({time.time() - t0:.0f}s, {n_q} questions)", flush=True)
    print(f"wrote {n_q} teacher questions -> {out} ({time.time() - t0:.0f}s, {dropped} dropped)",
          flush=True)


if __name__ == "__main__":
    main()
