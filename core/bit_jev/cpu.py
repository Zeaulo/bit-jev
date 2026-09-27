"""使用 bitnet.cpp CPU runner 对 TypeSafe 请求执行 bit-jev 分类。"""

import argparse
import json
import subprocess
from pathlib import Path

from .api import to_answers, to_record
from .model import encode, load_tokenizer


def prepare_request(request, tokenizer):
    """将共用 state 的请求改写为逐题因果行，并标记指针读取位置。"""
    # 原有 API 负责请求校验与 Noul、Choice、Score 选项展开。
    record, metadata = to_record(request, labelled=False)
    encoded = encode(tokenizer, record)
    state_len = encoded["seg"].count(0)
    rows = []
    branch_start = state_len
    # 每道题复用相同 state，并只接上本题分支以保持隔离。
    for decide, options in zip(encoded["decide_idx"], encoded["opt_idx"]):
        ids = encoded["ids"][:state_len] + encoded["ids"][branch_start:decide + 1]
        rows.append({
            "ids": ids,
            "decide": len(ids) - 1,
            "options": [state_len + option - branch_start for option in options],
        })
        branch_start = decide + 1
    return {"rows": rows}, metadata


def run_requests(requests, run, artifact, binary, threads=4, batch=256):
    """保持一个原生进程和模型常驻内存，逐条发送预编码请求。"""
    # tokenizer 与训练、评估使用同一个蒸馏运行目录。
    tokenizer = load_tokenizer(run)
    artifact = Path(artifact)
    command = [str(binary), "--model", str(artifact / "backbone-i2_s.gguf"),
               "--head", str(artifact / "head.f32"), "--threads", str(threads),
               "--batch", str(batch)]
    # 原生程序的 stderr 保持可见，stdout 只传输一行一个 JSON 结果。
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as process:
        try:
            # 输入记录顺序与结果顺序必须严格一致。
            for request in requests:
                prepared, metadata = prepare_request(request, tokenizer)
                process.stdin.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                process.stdin.flush()
                response_line = process.stdout.readline()
                if not response_line:
                    raise RuntimeError(f"CPU runner 在处理记录时退出：{process.poll()}")
                response = json.loads(response_line)
                logits = response["logits"]
                probabilities = response["probabilities"]
                if len(logits) != len(metadata) or len(probabilities) != len(metadata):
                    raise RuntimeError("CPU runner 的题目数量与输入不一致")
                # 复用既有答案格式，附加 raw logits 供数值对照。
                yield {
                    "record": request.get("_meta", {}).get("id"),
                    "answers": to_answers(probabilities, metadata),
                    "logits": logits,
                    "probabilities": probabilities,
                    "latency_ms": response["latency_ms"],
                }
        finally:
            # 关闭输入让原生程序在最后一条记录后正常退出。
            process.stdin.close()
            process.wait()


def main(argv=None):
    """从 JSONL 输入读取请求并保存 CPU 分类结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)
    # JSONL 逐行读入；limit=0 表示处理完整输入文件。
    def requests():
        """按文件顺序生成需要分类的请求。"""
        with open(args.input, encoding="utf-8") as source:
            for index, line in enumerate(source):
                if args.limit and index >= args.limit:
                    break
                if line.strip():
                    yield json.loads(line)
    # 输出文件统一采用 UTF-8 和 LF，便于与评估产物逐行比对。
    with open(args.out, "w", encoding="utf-8", newline="\n") as target:
        for result in run_requests(requests(), args.run, args.artifact, args.binary,
                                   threads=args.threads, batch=args.batch):
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()


if __name__ == "__main__":
    main()
