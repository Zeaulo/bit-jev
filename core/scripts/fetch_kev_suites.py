"""Materialize the kev eval suites bit-jev trains and benchmarks on.

    python scripts/fetch_kev_suites.py

Files under data/: data/decision-v7/{manifest,development,calibration,test,train}.{json,jsonl}
                   data/transfer-v4/{manifest,development,calibration,test,train}.{json,jsonl}

The pretrained kev authors publish decision-v7's training partition on the Hub (it is not in git);
this script downloads it through `bit_jev.hub` so HF and ModelScope both work. On AutoDL there is
no kev-suites mirror on ModelScope; copy data/{decision-v7,transfer-v4} onto the instance with
scp/rsync from the workstation (scripts/deploy_ssh.ps1 already pushes core/, so add data/ to it)
rather than expecting this script to succeed there.
"""
import argparse
import shutil
import sys
from pathlib import Path

# kev-suites' Hugging Face layout is by-version directory: v7/decision-v7/, v4/transfer-v4/, etc.
# (older suites sit at the top level: decision-v1/, smoke-v1/, external/semif-v1/, ...).
# manifest.json is also hostad, so we fetch it with the partitions.
FILES = ("manifest.json", "development.jsonl", "calibration.jsonl", "test.jsonl", "train.jsonl")
SUITES = ("v7/decision-v7", "v4/transfer-v4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="jaredpalmer/kev-suites")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bit_jev import hub

    out = Path(args.out)
    for suite in SUITES:
        local = out / Path(suite).name
        local.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            p = Path(hub.dataset_file(args.repo_id, f"{suite}/{f}"))
            shutil.copy2(p, local / f)
            print(f"  {local / f}")
    print(f"suites in {out}")


if __name__ == "__main__":
    main()
