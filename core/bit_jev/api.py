"""TypeSafe-compatible request/response shapes (same contract as kev.api).

Noul   -> 2 options [false, true];             answer = p(true)
Choice -> options 'name' or 'name: desc';      answer = argmax + probabilities + confidence
Score  -> options = ordered level descriptions; answer = expected level + probabilities + confidence
"""
import json
from typing import Any, Literal, Union

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
QUESTION_TYPES = ("noul", "choice", "score")


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list:
    """The keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def validate_request(req: dict) -> None:
    """Raise ValueError on a malformed SystemOne-shaped request dict."""
    if not isinstance(req, dict) or "questions" not in req:
        raise ValueError("request needs a 'questions' object")
    if not req["questions"]:
        raise ValueError("at least one question is required")
    for qid, q in req["questions"].items():
        t = q.get("type")
        if t not in QUESTION_TYPES:
            raise ValueError(f"question {qid!r}: type must be one of {QUESTION_TYPES}")
        if t in ("choice", "score"):
            crit = q.get("criteria")
            n = len(crit) if crit else 0
            if not 1 <= n <= MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: criteria must have 1..{MAX_OPTIONS} options")


def to_record(req: dict, labelled: bool = True):
    """SystemOne-shaped dict -> (internal record, per-question meta).

    With labelled=False, labels are zero placeholders and may be absent."""
    validate_request(req)
    qs, meta = [], []
    for qid, q in req["questions"].items():
        t = q["type"]
        m = {"id": qid, "type": t, "keys": question_keys(t, q.get("criteria"))}
        if t == "noul":
            c = q.get("criteria") or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            label = int(q["label"]) if "label" in q else -1
        elif t == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
            label = m["keys"].index(q["label"]) if "label" in q else -1
        else:
            opts = [render(x) for x in q["criteria"]]
            m["legend"] = dict(zip(m["keys"], opts))
            label = int(q["label"]) if "label" in q else -1
        if labelled and label < 0:
            raise ValueError(f"question {qid!r} of type {t} is missing a label")
        qs.append({"instr": render(q.get("instructions")), "options": opts,
                   "label": max(label, 0), "qtype": t})
        meta.append(m)
    return {"state": render(req.get("state")), "questions": qs}, meta


def load_requests(path):
    """Labelled requests from a JSONL file: one SystemOne-shaped object per line, a label per question."""
    reqs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
            r.setdefault("_meta", {})
            r["_meta"].setdefault("id", f"{path}:{i}")
            r["_meta"].setdefault("source", "user")
            reqs.append(r)
    if not reqs:
        raise ValueError(f"{path}: no records found")
    return reqs


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def choice_confidence(p) -> float:
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p) -> float:
    """1 - E|level - mode| / (L - 1): how concentrated the distribution is at its modal level."""
    L = len(p)
    if L == 1:
        return 1.0
    mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def round_prob(x: float) -> float:
    return round(float(x), 4)


def to_answers(probs, meta) -> dict:
    """Per-question probability lists -> TypeSafe-shaped answers (kev.api.to_answers)."""
    out = {}
    for p, m in zip(probs, meta):
        p = [float(x) for x in p]
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            out[m["id"]] = {
                "type": "choice",
                "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                "confidence": round_prob(choice_confidence(p)),
                "probabilities": {k: round_prob(v) for k, v in zip(m["keys"], p)},
            }
        else:
            out[m["id"]] = {
                "type": "score",
                "score": round_prob(sum(i * pi for i, pi in enumerate(p))),
                "confidence": round_prob(score_confidence(p)),
                "legend": m["legend"],
                "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
            }
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style figure: tokens of the serialized answers (there is no generation)."""
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
