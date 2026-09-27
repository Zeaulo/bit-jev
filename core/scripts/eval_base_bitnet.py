"""Zero-shot baseline: feed decision-v7 dev rows to the raw BitNet backbone and let it generate
JSON answers, with no Kev encoder/mask/head. Verifies whether near-random bit-jev accuracy is
caused by the backbone or by the LoRA+PointerHead training path.

    python scripts/eval_base_bitnet.py                  # 200 balanced rows on CPU (slow, no cost)
    python scripts/eval_base_bitnet.py --device cuda    # same on a card, ~10 min
    python scripts/eval_base_bitnet.py --limit 50       # subset

Answer contract per question:
  noul    -> true / false              (compared with the boolean label)
  choice  -> option key                (must be one of the criteria keys)
  score   -> level index as string     ("0" .. "K-1")
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # so ../learning/ resolves

os.environ.setdefault("BIT_JEV_HUB", "hf")

from bit_jev import DEFAULT_BASE, hub
from bit_jev.api import question_keys, render


def build_prompt(req):
    """One record -> one generation prompt. No delimiters, no mask -- natural language only."""
    state = render(req["state"])
    qids = list(req["questions"].keys())
    expected_keys = ", ".join(f'"{k}"' for k in qids)
    q_blocks = []
    for qid, q in req["questions"].items():
        t = q["type"]
        keys = question_keys(t, q.get("criteria"))
        instr = render(q.get("instructions") or "") or "(answer this question)"
        opt_options = " | ".join(keys)
        if t == "noul":
            option_spec = '"true" | "false"'
        elif t == "choice":
            option_spec = f'"{opt_options}"'
        else:
            option_spec = f'"0" | "1" ... | "{len(keys) - 1}"'
        q_blocks.append(f'  "{qid}": {instr}  (answer one of: {option_spec})')
    qs = "\n".join(q_blocks)
    return (
        "You are a strict decision model. Read the state and answer every question.\n"
        "Reply with ONLY a single valid JSON object, no prose, no markdown, no explanation, no newlines before or after the JSON.\n"
        f"The JSON object MUST have EXACTLY these {len(qids)} keys: {expected_keys}.\n"
        "Each value must be one of the allowed options, converted to a string.\n\n"
        f"State:\n{state}\n\n"
        f"Questions:\n{qs}\n\n"
        f"JSON (with exactly the keys {expected_keys}):"
    )


KEY_RE = re.compile(r'"([^"\\]+)"\s*:\s*("[^"\\]*"|true|false|null|-?\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_json_answers(text, req):
    """Best-effort JSON-object extraction from free-form LLM output.

    The model emits its answer first and then often loops repetitions; take the FIRST complete
    {...} object, not the last."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:end + 1])
                        return {str(k): str(v).strip() for k, v in obj.items()
                                if not isinstance(v, (dict, list))}
                    except json.JSONDecodeError:
                        break
    return {m.group(1): m.group(2).strip().strip('"') for m in KEY_RE.finditer(text)}


def coerce_answer(qtype, raw, keys):
    """Map a generated value to an option index; None when it cannot be parsed."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if qtype == "noul":
        if s in ("true", "yes", "1"): return 1
        if s in ("false", "no", "0"): return 0
        return None
    if qtype == "choice":
        for i, k in enumerate(keys):
            if s == k.lower(): return i
        for i, k in enumerate(keys):
            if k.lower() in s or s in k.lower(): return i
        return None
    m = re.match(r"^(\d+)$", s)
    if m and int(m.group(1)) < len(keys):
        return int(m.group(1))
    if s in keys:
        return keys.index(s)
    return None


def request_label(req, qid):
    q = req["questions"][qid]
    t = q["type"]
    keys = question_keys(t, q.get("criteria"))
    y = q["label"]
    if t == "choice":
        return keys.index(y)
    return int(y)


def sample_rows(rows, limit, seed):
    import random
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for i, r in enumerate(rows):
        for qid, q in r["questions"].items():
            by_type[q["type"]].append((i, qid))
    total_q = sum(len(v) for v in by_type.values())
    chosen = []
    for t, pool in by_type.items():
        k = max(1, round(limit * len(pool) / total_q))
        rng.shuffle(pool)
        chosen += pool[:k]
    rng.shuffle(chosen)
    return chosen[:limit]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "learning/kev/evals/v7/decision-v7/development.jsonl"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="", choices=["", "fp16", "bf16", "fp32"],
                    help="empty = fp16 on cuda (Turing cards have no bf16 tensor cores), fp32 on cpu")
    ap.add_argument("--single_q", type=int, default=1,
                    help="1: prompt only asks the sampled question (4x faster; the JSON then has one key)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "runs/base-bitnet-eval.json"))
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    total_q = sum(len(r["questions"]) for r in rows)
    print(f"loaded {len(rows)} records / {total_q} questions from {a.data}")

    chosen = sample_rows(rows, a.limit, a.seed)
    pairs = [(rows[i], i, qid) for i, qid in chosen]
    print(f"evaluating {len(pairs)} sampled answers on {a.device}")

    tok = hub.load_tokenizer(DEFAULT_BASE)
    tok.pad_token_id = tok.eos_token_id
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
                 "": torch.bfloat16 if a.device == "cuda" else torch.float32}
    dt = dtype_map[a.dtype]
    print(f"loading {DEFAULT_BASE} ({a.device}, dtype={dt}) ...")
    model = hub.load_model(DEFAULT_BASE, dtype=dt,
                           attn_implementation="eager", trust_remote_code=False)
    model.to(a.device).eval()
    # BitNet continuation tends to loop on "<|im|>" / "<|im_sep|>" repetitions; clamp the loop at
    # the JSON end marker or the EOS barrier, and never let generation run past 128 tokens.
    eos = tok.eos_token_id
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|end_of_text|>", "<|eot_id|>", "<|im|>", "<|im_sep|>") if tok.convert_tokens_to_ids(t) is not None and tok.convert_tokens_to_ids(t) >= 0]
    stop_ids = list({eos} | set(stop_ids))
    print(f"stop ids: {stop_ids}")

    n_scored = n_correct = 0
    per_q = Counter()
    details = []
    t0 = time.time()
    for idx, (req, i, qid) in enumerate(pairs):
        sub = req if a.single_q else req
        if a.single_q:
            sub = {"state": req["state"], "questions": {qid: req["questions"][qid]}}
        prompt = build_prompt(sub)
        inputs = tok(prompt, return_tensors="pt").to(a.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id,
                                 eos_token_id=stop_ids)
        new_text = tok.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answers = parse_json_answers(new_text, req)
        q = req["questions"][qid]
        keys = question_keys(q["type"], q.get("criteria"))
        y = request_label(req, qid)
        pred = coerce_answer(q["type"], answers.get(qid), keys)
        ok = pred is not None and pred == y
        n_scored += 1
        n_correct += ok
        per_q[q["type"]] += ok
        details.append({"record": i, "qid": qid, "qtype": q["type"], "y": y, "raw": answers.get(qid),
                        "pred": pred, "correct": ok,
                        "output": new_text[:200]})
        if (idx + 1) % 10 == 0 or idx == 0:
            acc = n_correct / n_scored
            print(f"  {idx + 1}/{len(pairs)}  acc={acc:.3f}  elapsed={time.time() - t0:.0f}s", flush=True)

    acc = n_correct / max(n_scored, 1)
    types = sorted({d["qtype"] for d in details})
    by_type = {}
    for t in types:
        sub = [d for d in details if d["qtype"] == t]
        by_type[t] = {"n": len(sub), "acc": sum(d["correct"] for d in sub) / len(sub)}
    report = {"base": DEFAULT_BASE, "data": a.data, "limit": a.limit, "seed": a.seed,
              "n_questions_scored": n_scored, "acc": acc, "by_type": by_type,
              "duration_s": time.time() - t0, "details": details[:50]}
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFINAL: acc={acc:.3f}")
    print(f"by_type: {json.dumps(by_type, indent=2)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
