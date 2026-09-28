"""把文档目录中的中英文模型卡转换为 Hub 仓库根目录的相对链接。"""

from pathlib import Path


# 发布目录本地保留 GGUF；本脚本只改写两份 README，不接触权重及哈希清单。
ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "test" / "release" / "hf-bit-jev-2b-distilled"
LINKS = {
    "HF_MODEL_CARD.zh-CN.md": "README.md",
    "HF_MODEL_CARD.md": "README.en.md",
    "figures/project-cover.zh-CN.png": "project-cover.zh-CN.png",
    "figures/project-cover.en.png": "project-cover.en.png",
    "figures/bit-jev-autodl-case.zh-CN.svg": "bit-jev-autodl-case.zh-CN.png",
    "figures/bit-jev-autodl-case.en.svg": "bit-jev-autodl-case.en.png",
}


def stage(source_name: str, target_name: str) -> None:
    """读取唯一的模型卡文档并生成 Hub 根目录对应的 README。"""
    content = (ROOT / "docs" / source_name).read_text(encoding="utf-8")
    # Hub 仓库的图片与双语 README 均位于根目录，不能保留 GitHub 的 figures 前缀。
    for old, new in LINKS.items():
        content = content.replace(old, new)
    (STAGING / target_name).write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    stage("HF_MODEL_CARD.zh-CN.md", "README.md")
    stage("HF_MODEL_CARD.md", "README.en.md")
