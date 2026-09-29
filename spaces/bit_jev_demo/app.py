"""ModelScope Docker Space 的薄入口；页面逻辑来自已发布的 bit-jev 包。"""

from __future__ import annotations

import os

from bit_jev.web.app import create_demo
from bit_jev.web.service import WebConfig


# 容器内编译好的 Linux CPU 程序由镜像提供，网页仍按需下载公开模型。
config = WebConfig(device="cpu", threads=min(os.cpu_count() or 2, 2),
                   source=os.environ.get("BIT_JEV_MODEL_SOURCE", "auto"),
                   binary=os.environ.get("BIT_JEV_BINARY", "/opt/bit-jev-cpu"),
                   public_space=True)
demo, session = create_demo(config)


if __name__ == "__main__":
    # 免费空间监听公开 7860 端口，单进程队列防止交错写入原生 JSONL。
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
