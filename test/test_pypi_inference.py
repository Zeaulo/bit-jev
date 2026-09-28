"""从公开模型包执行两次常驻 GGUF 推理，验证 pip 接口的真实调用。"""

import argparse
import json
import os
import sys
from pathlib import Path


# 从项目根目录运行本脚本时，优先测试当前待发布的 core 源码。
ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("BIT_JEV_TEST_INSTALLED"):
    sys.path.insert(0, str(ROOT / "core"))

from bit_jev.gguf import BitJev  # noqa: E402


def main():
    """加载模型一次、重复计算两次，并输出结构化校验结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args()
    # 手写输入只包含客服情境，不包含 Yelp 训练样本或隐私数据。
    request = json.loads((ROOT / "test" / "example_cpu_request.jsonl").read_text(encoding="utf-8").splitlines()[0])
    with BitJev.from_pretrained(args.model, binary=args.binary, device=args.device,
                                threads=args.threads, batch=128,
                                gpu_index=args.gpu_index) as model:
        results = list(model.infer_many((request, request)))
        if len(results) != 2 or results[0]["answers"] != results[1]["answers"]:
            raise AssertionError("常驻推理两次结果不一致")
        if not results[0]["answers"] or any(result["latency_ms"] <= 0 for result in results):
            raise AssertionError("缺少结构化答案或原生计时")
        print(json.dumps({"device": model.device, "answers": results[0]["answers"],
                          "latency_ms": [result["latency_ms"] for result in results]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
