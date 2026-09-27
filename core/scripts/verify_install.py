"""Final install check for bit-jev.

    python scripts/verify_install.py

Validates: module imports, the kev-format smoke data parses, delimiter ids resolve on a live
BitNet bf16 download (when `transformers` is installed and the network allows), the encoder
packs the README example, and every script referenced in README exists.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  ok  {name}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")
        failures.append((name, e))


def imports():
    import bit_jev  # noqa
    from bit_jev import api, checkpoint, convert, eval as be, model, serve, train  # noqa


def smoke_data():
    from bit_jev.api import load_requests, to_record
    reqs = load_requests(ROOT / "data" / "smoke.jsonl")
    assert len(reqs) >= 10, f"smoke.jsonl has {len(reqs)} records"
    for r in reqs:
        rec, meta = to_record(r, labelled=True)
        assert rec["questions"], r["_meta"]


def api_request():
    from bit_jev.api import to_record
    req = json.loads((ROOT / "data" / "api_request.json").read_text(encoding="utf-8"))
    to_record(req, labelled=False)


def tokenizer():
    from bit_jev.model import delimiter_ids, load_tokenizer, encode
    from bit_jev.api import to_record
    import json as _json
    tok = load_tokenizer(os.environ.get("BIT_JEV_BASE", "microsoft/bitnet-b1.58-2B-4T-bf16"))
    dids = delimiter_ids(tok)
    assert len(set(dids)) == 5, dids
    req = _json.loads((ROOT / "data" / "api_request.json").read_text(encoding="utf-8"))
    rec, _ = to_record(req, labelled=False)
    enc = encode(tok, rec)
    assert enc["ids"][0] == dids[0]


def scripts_exist():
    for s in ("setup_autodl.sh", "run_train.sh", "run_eval.sh", "run_convert.sh",
              "run_serve.sh", "oracle_bitnet.py", "fetch_kev_suites.py",
              "deploy_ssh.ps1", "pull_results.ps1", "smoke_train.sh"):
        assert (ROOT / "scripts" / s).exists(), s


def main():
    run = [("imports", imports), ("smoke data", smoke_data), ("api request", api_request),
           ("scripts exist", scripts_exist)]
    if os.environ.get("BIT_JEV_ONLINE", ""):
        run.append(("tokenizer + encode", tokenizer))
    else:
        print("  ok  (skipping tokenizer check; set BIT_JEV_ONLINE=1 to download the base)")
    for name, fn in run:
        check(name, fn)
    if failures:
        print(f"\n{len(failures)} FAILURES")
        sys.exit(1)
    print("\nall good. Next on GPU: bash scripts/smoke_train.sh")


if __name__ == "__main__":
    main()
