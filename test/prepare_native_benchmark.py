"""将 TypeSafe 请求编码为原生 runner 可直接读取的固定 JSONL。"""

import argparse
import json
import sys
from pathlib import Path


# 根目录测试脚本复用正式 CPU 请求编码，避免手写 token 协议。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bit_jev.cpu import prepare_request  # noqa: E402
from bit_jev.model import load_tokenizer  # noqa: E402


def prepare(run, input_path, output_path, limit, repeats):
    """按原始顺序写出请求，可重复多轮用于常驻模型测速。"""
    tokenizer = load_tokenizer(run)
    prepared = []
    with Path(input_path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            request = json.loads(line)
            encoded, _ = prepare_request(request, tokenizer)
            prepared.append(encoded)
            if limit and len(prepared) >= limit:
                break
    if not prepared or repeats < 1:
        raise ValueError("原生测速必须包含请求且重复次数至少为 1")
    with Path(output_path).open("w", encoding="utf-8", newline="\n") as target:
        for repeat_index in range(repeats):
            for record in prepared:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
    questions = sum(len(record["rows"]) for record in prepared)
    print(json.dumps({"records_per_round": len(prepared), "questions_per_round": questions,
                      "repeats": repeats, "output": str(output_path)}, ensure_ascii=False))


def main():
    """解析检查点、输入和重复轮次。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    prepare(args.run, args.input, args.out, args.limit, args.repeats)


if __name__ == "__main__":
    main()
