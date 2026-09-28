"""安装后可直接执行的结构化决策示例。"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .gguf import BitJev, DEFAULT_MODEL


# 示例只包含公开的手写客服场景，不携带训练记录或预先编造的模型答案。
DEMO_REQUEST: dict[str, Any] = {
    "state": "客户报告同一订单被重复扣款。",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "哪个团队应该处理？",
            "criteria": {"billing": "支付与退款", "shipping": "物流配送"},
        },
    },
}


def main(argv: list[str] | None = None) -> None:
    """下载或复用模型，运行一条请求并把真实答案打印为 JSON。"""
    # 命令行参数允许用户复用本地目录，并选择预编译 CPU 或 Vulkan 后端。
    parser = argparse.ArgumentParser(description="bit-jev 安装即用示例：首次运行将下载约 1.19 GB 模型")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 仓库或本地模型目录")
    parser.add_argument("--device", choices=("cpu", "gpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-index", type=int, help="GPU 后端的可见设备序号")
    args = parser.parse_args(argv)
    # 首次使用会按需下载权重；常驻模型完成加载后才运行示例题目。
    with BitJev.from_pretrained(args.model, device=args.device, threads=args.threads,
                                gpu_index=args.gpu_index) as model:
        result = model.infer(DEMO_REQUEST)
    # 输出实际推理的答案与原生耗时，不预置预测结果。
    print(json.dumps({"answers": result["answers"], "latency_ms": result["latency_ms"],
                      "device": result["device"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
