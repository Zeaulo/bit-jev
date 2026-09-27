"""Trained checkpoints: a directory (or Hub repo) holding a LoRA adapter, `head.pt` and the tokenizer.

    ck = BitJevCheckpoint("runs/bit-jev-2b")        # or a hub id
    tok, model = ck.load("cuda", dtype=torch.float32)

`head.pt` carries the architecture (Meta) and the pointer-head state dict; the load path is the only
place that knows both, so train / eval / serve / convert all build the same model.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import hub as _hub
from .model import DecisionModel, load_tokenizer

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local directory as given, or a Hub repo id (optionally @revision) downloaded to the HF cache."""
    if os.path.isdir(run):
        return str(run)
    repo, _, revision = str(run).partition("@")
    return _hub.snapshot(repo, revision=revision or None,
                         allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "*.model"])


@dataclass
class Meta:
    """Contents of head.pt. `extra` keeps everything else in the file (training args, hashes, fits)."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    temperature: float = 1.0
    arch: str = "bit-jev/0.1"
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "temperature", "arch", "holdout")

    @classmethod
    def from_dict(cls, d):
        if d.get("arch") not in (None, "bit-jev/0.1"):
            raise ValueError(f"head.pt arch {d.get('arch')!r} is not a bit-jev checkpoint")
        return cls(**{k: d[k] for k in cls.KNOWN if k in d},
                   extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu", weights_only=False))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


class BitJevCheckpoint:
    def __init__(self, run):
        self.requested = str(run)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def load(self, device, dtype=torch.float32, merge=None, temperature=None):
        """-> (tokenizer, model) in eval mode.

        merge=None defaults by base: on a BitNet base the LoRA adapter must stay UNMERGED. The
        online quantizer (transformers' bitnet integration) rounds every weight to {-1,0,+1} at
        each forward, so a merged delta is quantized away and the served model silently loses
        everything the adapter learned (73.1% unmerged vs 32.4% merged on decision-v7 dev).
        An unmerged LoRA side-branch computes in full precision and reproduces training exactly.
        merge=True remains available for dense fp bases (kev-style), where merging is exact.

        Distilled runs (bit_jev.distill) store a full backbone instead of an adapter: a run dir
        with config.json and no adapter_model.safetensors is loaded as the backbone itself."""
        has_adapter = self.file("adapter_model.safetensors").exists()
        meta = self.meta
        if has_adapter or not self.file("tokenizer_config.json").exists():
            tok = load_tokenizer(meta.base, revision=meta.base_revision)
        else:
            tok = load_tokenizer(self.path)   # distilled run: the run dir carries its own tokenizer
        if has_adapter:
            backbone = meta.base
        else:
            if not self.file("config.json").exists():
                raise FileNotFoundError(f"{self.path} has neither an adapter nor a full backbone "
                                        "(config.json missing); not a bit-jev run dir")
            backbone = self.path
        m = DecisionModel(backbone, tok, device, lora=None,
                          revision=meta.base_revision if has_adapter else None,
                          head_dim=meta.head_dim, dtype=dtype)
        if has_adapter:
            from peft import PeftModel
            m.lm = PeftModel.from_pretrained(m.lm, self.path, torch_device=str(device)).to(device)
            if merge is None:
                merge = not self._is_bitnet_base()
            if merge:
                m.lm = m.lm.merge_and_unload()
        if dtype != torch.float32:
            m.lm = m.lm.to(dtype)
        m.head.load_state_dict(meta.head)
        m.head.temperature = meta.temperature if temperature is None else float(temperature)
        m.eval()
        return tok, m

    def _is_bitnet_base(self):
        """True when the checkpoint's base uses BitNet's online ternary quantization."""
        if "bitnet" in (self.meta.base or "").lower():
            return True
        try:
            from . import hub as _hub
            cfg = _hub.load_config(self.meta.base, revision=self.meta.base_revision)
            return getattr(cfg, "model_type", "") == "bitnet"
        except Exception:
            return False

    COMPAT_FIELDS = ("base", "lora", "head_dim")

    def warm_start(self, model, ours: Meta):
        """Delta training: load this checkpoint's adapter and head into a fresh DecisionModel;
        architecture fields are compared before loading (a silent mismatch still "trains" otherwise)."""
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        for name in self.COMPAT_FIELDS:
            if getattr(self.meta, name) != getattr(ours, name):
                raise ValueError(f"--init_from {self.path}: {name} differs "
                                 f"({getattr(self.meta, name)!r} vs {getattr(ours, name)!r})")
        weights = load_peft_weights(self.path, device="cpu")
        have = set(get_peft_model_state_dict(model.lm))
        if set(weights) != have:
            raise ValueError(f"--init_from {self.path}: adapter tensors mismatch "
                             f"(missing {sorted(have - set(weights))[:2]}, "
                             f"unexpected {sorted(set(weights) - have)[:2]})")
        set_peft_model_state_dict(model.lm, weights)
        model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "adapter_tensors": len(weights)}
