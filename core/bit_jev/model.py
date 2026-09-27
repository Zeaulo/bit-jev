"""Decision model on a 1.58-bit BitNet backbone: prefill-only, block-causal branch mask, pointer readout.

Same construction as kev.model, with three BitNet-specific choices:

1. Delimiters are BitNet's own `<|reserved_special_token_0|>` .. `_4|>` (ids 128002, 128003, 128004,
   128005, 128008 under the official microsoft/AI-ModelScope tokenizer). The HF/ModelScope bf16
   checkpoint ships exactly-ternary embedding tables, so fine-tuning never has to resize them;
   LoRA learns the masking semantics from scratch. The Qwen tokens kev uses (`<|fim_prefix|>` etc.)
   are not in the BitNet vocab.
2. The user-text delimiter scrubber also matches BitNet's `<|reserved_special_token_N|>` pattern, so
   callers cannot forge option/branch boundaries (kev rewrites `<|name|>` -> `<¦name¦>`; same here).
3. The backbone is attention-only (no DeltaNet layers), so every form runs the packed block-causal
   mask; the row form exists only as a memory fallback for very long serving inputs.

The LM head is never used: the loss and the readout come from `last_hidden_state` at the option
boundary tokens only.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hub as _hub
from .hub import load_tokenizer as _load_tokenizer

SPECIAL_TOKENS = ["<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
                  "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
                  "<|reserved_special_token_4|>"]
# ids under the microsoft/bitnet-b1.58-2B-4T-bf16 tokenizer, for reference only; delimiter_ids()
# resolves them through the tokenizer at runtime (each SPECIAL token is its own id, consecutive
# in the reserved range; the assert in delimiter_ids fails hard if this changes under us).
DELIMITER_HINTS = {
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|reserved_special_token_2|>": 128004,
    "<|reserved_special_token_3|>": 128005,
    "<|reserved_special_token_4|>": 128008,
}

# training context: state tokens, tokens per question branch, whole packed record
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# serving context: same upper bounds kev uses; longer than training by design
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 8192, 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH

OPT_NONE, OPT_DECIDE = -1, -2

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflow(ValueError):
    """A request does not encode within its context (state, branch or packed limit)."""


def load_tokenizer(name, revision=None):
    return _load_tokenizer(name, revision=revision)


def pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def delimiter_ids(tok):
    """Delimiter token ids, in the conventional order <state> <q> <opt> </opt> <decide>."""
    out = []
    for t in SPECIAL_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if i is None or i < 0:
            i = DELIMITER_HINTS[t]
        if not tok.convert_ids_to_tokens(i) == t:
            raise ValueError(f"{t} is not a single token in this tokenizer ({tok.name_or_path}); "
                             "bit-jev needs the five reserved special tokens of the BitNet vocab")
        out.append(i)
    return out


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt> ... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after the state),
    decide_idx [Q], opt_idx [Q][K] (index of each option's closing token), and per-token `opt`.
    Same layout as kev.encode, so records, training data and suites are interchangeable.
    """
    state_toks = user_tokens(tok, rec["state"])
    if strict and len(state_toks) + 1 > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_toks) + 1}")
    S = [delimiter_ids(tok)[0]] + state_toks[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    _, q_id, o_id, c_id, d_id = delimiter_ids(tok)
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state "
                                  f"(row limit {max_branch})")
        base = len(ids); p0 = len(S)
        ids += br; seg += [k] * len(br); pos += list(range(p0, p0 + len(br)))
        opt += [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]],
            "state_truncated": len(state_toks) + 1 > max_state}


def fits(rec, tok, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    try:
        return len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed
    except ValueError:
        return False


def branch_mask_batch(segs, device, dtype=torch.float32, length=None):
    """attend(i, j) iff j <= i and (seg[j] == 0 or seg[j] == seg[i]). Additive [B, 1, L, L].

    Right-padded to the longest sequence; padded keys are masked for every query, padded query rows
    keep the diagonal so no row is fully masked (finfo.min, not -inf, so softmax stays finite)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


class PointerHead(nn.Module):
    """Scores each option's closing-token hidden state against the question's <decide> hidden state."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # calibration: eval-mode logits are divided by this (1.0 = raw). The checkpoint carries a value
        # fitted on the development rows (calibrate_checkpoint.py); training always runs T=1 and the
        # argmax is unchanged by construction.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K, d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


class DecisionModel(nn.Module):
    """BitNet backbone (no LM head) + LoRA + PointerHead, prefill-only under the branch mask."""

    def __init__(self, name, tok, device, lora=None, revision=None, attn=None,
                 head_dim=256, dtype=torch.float32):
        super().__init__()
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        # transformers 5.13 registers bitnet natively. The checkpoint's config.json still carries
        # an auto_map that triggers resolve_trust_remote_code's SIGALRM path, which is POSIX-only
        # and crashes on Windows before any download starts; passing trust_remote_code=False tells
        # transformers to use its registered definition instead of looking up custom code.
        self.lm = _hub.load_model(name, revision=revision, dtype=dtype,
                                  attn_implementation=attn, trust_remote_code=False).model
        self.pad_id = pad_id(tok)
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora,
                             lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    backend = "torch"

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, **kw)

    # -- packed forward ---------------------------------------------------------------

    def _pad_rows(self, rows):
        """Right-pad (ids, pos) token rows into [N, L] tensors plus a [N, L] attention mask."""
        L = max(len(ids) for ids, _ in rows)
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device)
            pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device)
            att[i, : len(rid)] = 1
        return ids, pos, att

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a padded batch of packed records under the branch mask."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=dt, length=ids.shape[1])
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)])
                for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    # -- row form (memory fallback for very long serving inputs) -------------------------

    def rows_form(self, encs):
        return any(len(e["ids"]) > SERVE_MAX_PACKED for e in encs)

    def _rows_hidden(self, rows):
        out = []
        for rid, rpos in rows:
            ids = torch.tensor([rid], device=self.device)
            pos = torch.tensor([rpos], device=self.device)
            out.append(self.lm(input_ids=ids, position_ids=pos).last_hidden_state[0].float())
        return out

    def forward_rows_batch(self, encs):
        """Pack-equivalent: one causal row per question. Recomputes the state per question; exact."""
        out = [[] for _ in encs]
        for b, e in enumerate(encs):
            Ls = e["seg"].count(0)
            rows, readouts = [], []
            start = Ls
            for k, (d, oi) in enumerate(zip(e["decide_idx"], e["opt_idx"]), start=1):
                end = d + 1
                rows.append((e["ids"][:Ls] + e["ids"][start:end], e["pos"][:Ls] + e["pos"][start:end]))
                readouts.append((len(rows) - 1, Ls + d - start, [Ls + o - start for o in oi]))
                start = end
            hs = self._rows_hidden(rows)
            for row, d, oi in readouts:
                out[b].append(self.head(hs[row][d], hs[row][torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs):
        """List (per record) of lists (per question) of logits."""
        if self.rows_form(encs):
            return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
