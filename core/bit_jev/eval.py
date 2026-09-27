"""Benchmark a bit-jev checkpoint on labelled records (kev's accuracy / Brier / NLL / ECE contract).

    python -m bit_jev.eval --run runs/bit-jev-2b --data dev.jsonl --out runs/bit-jev-2b-dev
    python -m bit_jev.eval --run runs/bit-jev-2b --suite evals/v4/transfer-v4 --suite_split development --out runs/t4

Writes rows.jsonl (one row per (record, question), with logits) and report.json (accuracy, Brier,
NLL, 10-bin ECE, per-question-type breakdown). `--fit_temperature` fits a single global temperature
on the scored rows (what a released checkpoint carries); --write_back stores it into head.pt.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .api import read_jsonl, to_record, write_json, write_jsonl
from .checkpoint import BitJevCheckpoint


def argmax(a):
    return max(range(len(a)), key=lambda i: a[i])


def softmax_np(z):
    z = np.asarray(z, dtype=np.float64); z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score_rows(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    brier = nll = 0.0
    for r in rows:
        p = np.asarray(r["probs"], dtype=np.float64)
        t = np.zeros_like(p); t[r["label"]] = 1.0
        brier += float(((p - t) ** 2).sum())
        nll += float(-math.log(max(p[r["label"]], 1e-12)))
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        c = max(r["probs"]); b = min(int(c * 10), 9)
        bins[b][0] += 1; bins[b][1] += c; bins[b][2] += r["correct"]
    ece = sum(cnt / n * abs(corr / cnt - conf / cnt) for _, (cnt, conf, corr) in bins.items() if cnt)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["qtype"]].append(r)
    by = {t: {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs)} for t, rs in by_type.items()}
    return {"n": n, "acc": acc, "brier": brier / n, "nll": nll / n, "ece": ece, "by_type": by}


def fit_temperature(rows):
    """Single global temperature minimizing NLL over all rows (the kev release knob; argmax-safe)."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 6.0, 111):
        nll = 0.0
        for r in rows:
            p = softmax_np(np.asarray(r["logits"]) / t)
            nll -= math.log(max(p[r["label"]], 1e-12))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


@torch.no_grad()
def score(model, tok, reqs):
    rows = []
    for i, req in enumerate(reqs):
        rec, meta = to_record(req, labelled=True)
        enc = model.encode(tok, rec)
        logits_list = model.forward(enc)
        for q, m, z in zip(rec["questions"], meta, logits_list):
            zl = z.float().cpu().tolist()
            p = F.softmax(z.float(), -1).cpu().tolist()
            y = q["label"]
            rows.append({"record": req["_meta"]["id"], "qid": m["id"], "qtype": m["type"],
                         "keys": m["keys"], "label": y, "pred": argmax(p),
                         "probs": p, "logits": zl, "correct": int(argmax(p) == y)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(reqs)}", flush=True)
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bit_jev.eval")
    ap.add_argument("--run", required=True, help="a trained run dir or hub id")
    ap.add_argument("--data", default="", help="labelled JSONL (alternative to --suite)")
    ap.add_argument("--suite", default="", help="kev-format suite dir")
    ap.add_argument("--suite_split", default="development")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many records")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_merge", action="store_true",
                    help="keep the LoRA adapter unmerged (serving-style). With a BitNet base the "
                         "online quantizer otherwise rounds the merged delta away at every forward.")
    ap.add_argument("--raw", action="store_true", help="ignore the checkpoint's fitted temperature")
    ap.add_argument("--fit_temperature", action="store_true")
    ap.add_argument("--write_back", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0, help="override temperature")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.data) == bool(a.suite):
        ap.error("exactly one of --data / --suite is required")
    if a.write_back and not a.fit_temperature:
        ap.error("--write_back needs --fit_temperature")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = BitJevCheckpoint(a.run)
    dtype = torch.bfloat16 if a.bf16 else torch.float32
    temperature = 1.0 if a.raw else (a.temperature or None)
    tok, model = ck.load(dev, dtype=dtype, temperature=temperature, merge=not a.no_merge)
    model.eval()

    reqs = read_jsonl(Path(a.suite) / f"{a.suite_split}.jsonl") if a.suite else read_jsonl(a.data)
    for j, r in enumerate(reqs):
        r.setdefault("_meta", {}).setdefault("id", f"{a.data or a.suite}:{j}")
    if a.limit:
        reqs = reqs[: a.limit]
    print(f"scoring {len(reqs)} records on {dev} (dtype={str(dtype).replace('torch.','')}, T={model.head.temperature})", flush=True)

    rows = score(model, tok, reqs)
    report = score_rows(rows)
    report["temperature"] = model.head.temperature
    report["dtype"] = str(dtype).replace("torch.", "")
    report["run"] = a.run

    if a.fit_temperature:
        t = fit_temperature(rows)
        calibrated = [{**r, "probs": softmax_np(np.asarray(r["logits"]) / t).tolist()} for r in rows]
        report["temperature_fitted"] = t
        report["calibrated"] = score_rows(calibrated)
        if a.write_back:
            from .checkpoint import write_meta
            meta = ck.meta
            meta.temperature = t
            write_meta(ck.path, meta)
            print(f"fitted temperature {t:.3f} written back to {ck.path}/head.pt", flush=True)

    write_jsonl(out / "rows.jsonl", rows)
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
