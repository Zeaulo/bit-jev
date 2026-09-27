"""在相同请求上比较 I2_S 原生 CPU 路径与蒸馏 PyTorch 路径。"""

import argparse
import json
import os
import sys
from pathlib import Path

# Windows 的对照环境缺少可用的 Triton 编译器，保持参考路径为 eager 执行。
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import torch


# 让根目录测试脚本使用项目正式包，而不复制编码或打分逻辑。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bit_jev.api import to_record  # noqa: E402
from bit_jev.checkpoint import BitJevCheckpoint  # noqa: E402
from bit_jev.cpu import run_requests  # noqa: E402


def verify(run, artifact, binary, input_file, limit, threads, batch):
    """检查逐题候选数、logit 差异和最终选项是否一致。"""
    # 小样本对照使用正式请求格式；CPU 模型进程只启动一次。
    with open(input_file, encoding="utf-8") as source:
        requests = [json.loads(line) for line in source if line.strip()][:limit]
    cpu_results = list(run_requests(requests, run, artifact, binary, threads=threads, batch=batch))
    # 蒸馏检查点是原始在线量化路径，单独加载以防共享状态污染。
    checkpoint = BitJevCheckpoint(run)
    tokenizer, model = checkpoint.load("cpu", dtype=torch.float32)
    max_diff = 0.0
    flips = 0
    questions = 0
    with torch.no_grad():
        for request, cpu_result in zip(requests, cpu_results):
            record, _ = to_record(request, labelled=False)
            encoded = model.encode(tokenizer, record)
            reference = model.forward_rows_batch([encoded])[0]
            assert len(reference) == len(cpu_result["logits"])
            for expected, actual in zip(reference, cpu_result["logits"]):
                assert len(expected) == len(actual)
                actual_tensor = torch.tensor(actual)
                max_diff = max(max_diff, float((expected.cpu() - actual_tensor).abs().max()))
                flips += int(int(expected.argmax()) != int(actual_tensor.argmax()))
                questions += 1
    print(json.dumps({"records": len(requests), "questions": questions,
                      "max_logit_diff": max_diff, "argmax_flips": flips}, ensure_ascii=False))
    if flips:
        raise AssertionError(f"{flips} 道题的 CPU 与 PyTorch 选项不一致")


def main():
    """解析路径、对照规模和原生 CPU 运行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    verify(args.run, args.artifact, args.binary, args.input, args.limit, args.threads, args.batch)


if __name__ == "__main__":
    main()
