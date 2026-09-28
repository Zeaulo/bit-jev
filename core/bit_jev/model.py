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

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hub as _hub
from .encoding import (ContextOverflow, DELIMITER_HINTS, MAX_BRANCH, MAX_PACKED, MAX_STATE,
                       OPT_DECIDE, OPT_NONE, SERVE_MAX_BRANCH, SERVE_MAX_PACKED,
                       SERVE_MAX_STATE, SPECIAL_TOKENS, delimiter_ids, encode, fits,
                       load_tokenizer, pad_id, user_tokens)

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
