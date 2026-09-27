"""Export a trained bit-jev checkpoint to the bitnet.cpp CPU-inference layout (Path B artifact).

    python -m bit_jev.convert --run runs/bit-jev-2b --out export/bit-jev-2b

Produces:

    export/bit-jev-2b/
      merged/       full HF model: BitNet bf16 weights merged with the fp32 LoRA delta, tokenizer
      head.pt       pointer head + fitted temperature (kev's schema)
      pointer.json  delimiter ids and the readout formula a CPU runner needs
      README.md     how the pieces fit together

Post-export, scripts/oracle_bitnet.py asserts every merged weight is in {-1, 0, +1}, so the
following I2_S GGUF conversion (bitnet.cpp's own utils/convert-hf-to-gguf-bitnet.py) is lossless.
"""
import argparse
import json
from pathlib import Path

import torch

from .checkpoint import BitJevCheckpoint, write_meta
from .model import SPECIAL_TOKENS, delimiter_ids


README_NOTES = """# bit-jev export

Path-B CPU artifact. Three pieces:

1. **`merged/`** -- the full BitNet backbone with the LoRA delta merged in fp32. Every weight is
   still in {-1, 0, +1} (checked by scripts/oracle_bitnet.py), so bitnet.cpp's I2_S conversion
   is lossless. Feed this dir to utils/convert-hf-to-gguf-bitnet.py and then their quantize step.
2. **`head.pt`** -- the pointer head (Wq, Wk) and the fitted temperature. Runs alongside bitnet.cpp:
   take hidden states at every </opt> and at <decide>, score with the same formula kev uses.
3. **`pointer.json`** -- delimiter ids and the readout formula, so the CPU runner needs no Python
   copy of the training code to reproduce the answers.
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bit_jev.convert")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ck = BitJevCheckpoint(a.run)
    if "bitnet" in (ck.meta.base or "").lower():
        raise SystemExit(
            "refusing to export a merged BitNet checkpoint: the online ternary quantizer would "
            "round the merged LoRA delta away and the CPU model would serve near-random answers "
            "(32.4% vs 73.1% unmerged on decision-v7 dev). The CPU export needs a high-precision "
            "delta path (bitnet.cpp LoRA loading, or a Q8/Q4 quantized side-branch) -- see "
            "README 'Export contract'. Train/eval/serving with the unmerged adapter meanwhile.")
    tok, model = ck.load("cpu", dtype=torch.float32, merge=True)

    merged_dir = out / "merged"
    model.lm.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

    write_meta(out, ck.meta)

    dids = delimiter_ids(tok)
    names = ("state", "q", "opt", "close_opt", "decide")
    pointer = {
        "arch": ck.meta.arch,
        "base": ck.meta.base,
        "special_tokens": dict(zip(names, SPECIAL_TOKENS)),
        "delimiter_ids": dict(zip(names, dids)),
        "readout": "logits_k = (Wq h_<decide> / sqrt(d)) dot (Wk h_</opt>_k); softmax over options",
        "temperature": ck.meta.temperature,
    }
    (out / "pointer.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README_NOTES, encoding="utf-8")
    print(f"exported {ck.requested} -> {out}")
    print("next: python scripts/oracle_bitnet.py --run", a.run)


if __name__ == "__main__":
    main()
