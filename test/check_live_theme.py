"""检查在线 Space 是否实际加载当前包内的概率主题样式。"""

import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

# 线上配置必须与当前工作区源码一致，避免只验收旧容器。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bit_jev.web.theme import PAGE_CSS


def main() -> None:
    """读取公开配置与静态嵌入入口，不触发模型下载。"""
    response = requests.get("https://jinghao9616-bit-jev-demo.ms.show/config",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    if response.json().get("css", "").strip() != PAGE_CSS.strip():
        raise AssertionError("在线空间未加载当前主题样式")
    print("ModelScope 在线主题 CSS 与当前源码一致")
    # Hugging Face 入口通过 iframe 使用同一运行页面。
    info = HfApi().space_info("jinghao1632/bit-jev-demo")
    path = hf_hub_download("jinghao1632/bit-jev-demo", "index.html",
                           repo_type="space", revision=info.sha)
    if "jinghao9616-bit-jev-demo.ms.show" not in Path(path).read_text(encoding="utf-8"):
        raise AssertionError("Hugging Face 入口未嵌入已修复的运行页面")
    print("Hugging Face 已发布的静态入口指向同一运行页面", getattr(info, "host", None))


if __name__ == "__main__":
    main()
