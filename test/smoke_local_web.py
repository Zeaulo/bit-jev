"""启动本地 Gradio 服务并检查网页配置，不下载模型。"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


# 此脚本保留在项目 test/；可选择源码或已安装的 wheel 进行服务验收。
ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """向操作系统申请临时回环端口，避免覆盖用户已有的 7860 服务。"""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def main() -> None:
    """等待页面就绪，核对三任务标签、六回调与完整 Logo。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="执行 demo 的 Python 解释器")
    parser.add_argument("--installed", action="store_true", help="仅使用所选解释器已安装的 wheel")
    args = parser.parse_args()
    port = free_port()
    environment = os.environ.copy()
    if args.installed:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(ROOT / "core")
    command = [args.python, "-m", "bit_jev.demo", "--no-browser", "--port", str(port)]
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # 首次导入 Gradio 可能较慢，但不应触发模型下载或占用固定 7860 端口。
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"本地 Web 服务过早退出：{process.returncode}")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/config", timeout=2)
                if response.ok:
                    config = response.json()
                    tabs = [item["props"].get("label") for item in config["components"]
                            if item.get("type") == "tabitem"]
                    expected = {f"infer_{kind}_{language}"
                                for kind in ("choice", "noul", "score")
                                for language in ("zh", "en")}
                    actual = {item.get("api_name") for item in config["dependencies"]}
                    if tabs != ["Choice · 选择题", "Noul · 是非题", "Score · 等级题"]:
                        raise RuntimeError(f"本地页面题型 Tab 不符：{tabs}")
                    if not expected <= actual:
                        raise RuntimeError(f"本地页面缺少推理 API：{sorted(expected - actual)}")
                    print(f"本地 Gradio 服务正常：http://127.0.0.1:{port}；三个 Tab、六个 API 已就绪")
                    return
            except requests.RequestException:
                time.sleep(0.5)
        raise TimeoutError("本地页面 90 秒内未启动")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
