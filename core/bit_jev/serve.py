"""TypeSafe-compatible FastAPI server on a bit-jev checkpoint.

    BIT_JEV_API_KEY=secret python -m bit_jev.serve --run runs/bit-jev-2b --port 8009

    POST /v1/systemone          one forward pass, answers in kev's shape
    POST /v1/systemone/separate each question in its own pass
    GET  /v1/models             what is serving (device, dtype, temperature, base)
"""
import argparse
import os
import threading
import time
import uuid

import torch
import torch.nn.functional as F

from .api import output_tokens, to_answers, to_record
from .checkpoint import BitJevCheckpoint
from .model import ContextOverflow


def create_app(run, device, bf16):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as e:
        raise SystemExit("serve needs `pip install fastapi uvicorn`") from e

    ck = BitJevCheckpoint(run)
    dtype = torch.bfloat16 if bf16 else torch.float32
    tok, model = ck.load(device, dtype=dtype)
    model.eval()
    lock = threading.Lock()
    api_key = os.environ.get("BIT_JEV_API_KEY", "")
    app = FastAPI(title="bit-jev", version="0.1")

    async def auth(authorization: str | None = Header(default=None)):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "unauthorized")

    def answers_for(body):
        rec, meta = to_record(body, labelled=False)
        t0 = time.time()
        with lock, torch.no_grad():
            enc = model.encode(tok, rec)
            logits = model.forward(enc)
            probs = [F.softmax(z.float(), -1).cpu().tolist() for z in logits]
        return to_answers(probs, meta), int((time.time() - t0) * 1000)

    def envelope(body, answers, ms):
        return {"model": body.get("model", "bit-jev"),
                "answers": answers,
                "latency_ms": ms,
                "usage": {"output_tokens": output_tokens(tok, answers)},
                "x-typesafe-request-id": str(uuid.uuid4())}

    @app.post("/v1/systemone", dependencies=[Depends(auth)])
    async def systemone(body: dict):
        try:
            answers, ms = answers_for(body)
        except ContextOverflow as e:
            raise HTTPException(422, str(e)) from e
        return envelope(body, answers, ms)

    @app.post("/v1/systemone/separate", dependencies=[Depends(auth)])
    async def separate(body: dict):
        out, t_total = {}, 0
        for qid, q in body["questions"].items():
            one, ms = answers_for({"state": body.get("state"), "questions": {qid: q}})
            out.update(one)
            t_total += ms
        return envelope(body, out, t_total)

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "bit-jev", "object": "model"}],
                "device": device, "dtype": "bf16" if bf16 else "fp32",
                "temperature": model.head.temperature, "base": ck.meta.base}

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bit_jev.serve")
    ap.add_argument("--run", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = create_app(a.run, dev, a.bf16)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
