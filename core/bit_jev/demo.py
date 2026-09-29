"""安装后默认启动本地 Gradio 页面，也可运行一次旧式 JSON 示例。"""

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
    """解析页面启动配置；显式 --once 才执行原有单题命令模式。"""
    parser = argparse.ArgumentParser(description="bit-jev 本地 Gradio 页面：首次提交时下载约 1.19 GB 模型")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型仓库 ID 或本地模型目录")
    parser.add_argument("--source", choices=("auto", "huggingface", "modelscope"),
                        default="auto", help="模型下载来源；auto 在 Hugging Face 失败时回退 ModelScope")
    parser.add_argument("--device", choices=("cpu", "gpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-index", type=int, help="GPU 后端的可见设备序号")
    parser.add_argument("--binary", help="可选：已编译的原生程序路径")
    parser.add_argument("--host", default="127.0.0.1", help="页面监听地址；默认仅本机访问")
    parser.add_argument("--port", type=int, default=7860, help="页面端口，默认 7860")
    parser.add_argument("--no-browser", action="store_true", help="启动服务但不自动打开浏览器")
    parser.add_argument("--once", action="store_true", help="运行一次内置客服题并打印 JSON")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    if args.once:
        # 显式单题模式保留旧脚本的 JSON 输出，模型只在这里按需下载。
        with BitJev.from_pretrained(args.model, source=args.source,
                                    device=args.device, threads=args.threads,
                                    gpu_index=args.gpu_index, binary=args.binary) as model:
            result = model.infer(DEMO_REQUEST)
        print(json.dumps({"answers": result["answers"], "latency_ms": result["latency_ms"],
                          "device": result["device"]}, ensure_ascii=False, indent=2))
        return
    # 页面模块延迟导入；展示表单本身不会加载原生模型或下载权重。
    from .web.app import launch_local
    from .web.service import WebConfig

    config = WebConfig(model=args.model, source=args.source, device=args.device,
                       threads=args.threads, gpu_index=args.gpu_index, binary=args.binary)
    launch_local(config, host=args.host, port=args.port,
                 open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
