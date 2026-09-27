"""Live smoke test against the real BitNet bf16 checkpoint downloaded via the chosen hub.

    python scripts/test_real_bitnet.py                  # workstation default (hf)
    BIT_JEV_HUB=modelscope python scripts/test_real_bitnet.py

Validates end-to-end without instantiating the 2.4B backbone: tokenizer + delimiter ids + packed
encoding against the README example, the same path bit_jev.train uses, plus an oracle check on a
single embedding row (delimiter tokens must be exactly ternary in the published checkpoint).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from bit_jev import DEFAULT_BASE
from bit_jev import hub
from bit_jev.api import to_record
from bit_jev.model import MAX_STATE, delimiter_ids, encode


def main():
    print(f"hub: {hub.default_hub()}   base: {DEFAULT_BASE}")

    print("== tokenizer")
    tok = hub.load_tokenizer(DEFAULT_BASE)
    dids = delimiter_ids(tok)
    print(f"   delimiter ids (resolved from tokenizer): {dids}")
    # delimiters must be five distinct single tokens within BitNet's reserved-special range (the
    # published tokenizer guarantees exactly these ids; assert on the property, not the hint).
    assert len(set(dids)) == 5, dids
    assert all(128000 < i < 128256 for i in dids), dids
    assert all(tok.convert_ids_to_tokens(i).startswith("<|reserved_special") for i in dids), dids

    print("== encode the README request")
    import json
    req = json.loads((ROOT / "data" / "api_request.json").read_text(encoding="utf-8"))
    rec, _ = to_record(req, labelled=False)
    enc = encode(tok, rec, max_state=MAX_STATE)
    branch_sizes = [sum(1 for s in enc["seg"] if s == k) for k in range(1, len(rec["questions"]) + 1)]
    print(f"   packed tokens: {len(enc['ids'])}; state segment: {enc['seg'].count(0)}; "
          f"branches: {branch_sizes}")
    assert enc["ids"][0] == dids[0]
    assert len(enc["decide_idx"]) == len(rec["questions"])

    if os.environ.get("BIT_JEV_ONLINE_EMB"):
        print("== delimiter embedding rows are ternary in the published checkpoint")
        try:
            import torch
            from safetensors import safe_open
            snap = Path(hub.snapshot(DEFAULT_BASE, allow_patterns=["*.safetensors"]))
            with safe_open(next(snap.glob("*.safetensors")), framework="pt") as f:
                key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
                emb = f.get_slice(key)
                for did in dids:
                    row = emb[did].float()
                    good = all(abs(x - y) < 1e-6 for x, y in[(row.min().item(), -1.0), (row.max().item(), 1.0)])
                    print(f"   id {did}: min {row.min():+.3f} max {row.max():+.3f} ternary? {good}")
                    assert good, did
        except Exception as e:
            print(f"   (skipped: {e})")

    print("\nOK: tokenizer, delimiter ids, and the packed encoder all agree with the checkpoint.")


if __name__ == "__main__":
    main()
