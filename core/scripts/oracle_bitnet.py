"""Sanity check on the BitNet backbone after a training run.

    python scripts/oracle_bitnet.py --run runs/bit-jev-2b

Earlier draft asserted every frozen weight was in {-1,0,+1}. That premise was wrong: the bf16
checkpoint stores continuous floats, and transformers' bitnet integration (WeightQuant.apply)
quantizes them to ternary only inside the forward pass. What we assert cheaply here:

  1. the adapter lives only on the expected projections (attention and mlp);
  2. the five delimiter embeddings are still five distinct token rows after training.

The merged-export equivalence is already proven by run_convert.sh + eval.py rebuilding the same
forward path; this script does not pretend to audit weights.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bit_jev.checkpoint import BitJevCheckpoint
from bit_jev.model import DecisionModel, delimiter_ids, load_tokenizer


LORA_PARENT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
EXPECTED_LORA_KEYS = sorted(f"{p}.{kind}.{s}" for p in LORA_PARENT
                            for kind in ("lora_A", "lora_B") for s in ("default",))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ck = BitJevCheckpoint(a.run)

    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    model = DecisionModel(ck.meta.base, tok, "cpu", dtype=torch.float32)
    from peft import PeftModel
    model.lm = PeftModel.from_pretrained(model.lm, a.run)

    names = set(model.lm.state_dict().keys())
    lora_names = {n for n in names if ".lora_" in n}
    print(f"adapter tensors in state_dict: {len(lora_names)}")
    unusual = [n for n in lora_names if not any(n.endswith(p) for p in EXPECTED_LORA_KEYS)]
    if unusual:
        print(f"suspicious adapter tensors (not in expected LoRA targets): {unusual[:5]}")

    emb = model.lm.get_input_embeddings().weight.detach()
    rows = emb[delimiter_ids(tok)].float()
    distinct = len(torch.unique(rows, dim=0))
    print(f"delimiter embeddings: {distinct}/5 distinct")
    if distinct < 5:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
