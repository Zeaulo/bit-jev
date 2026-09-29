"""只同步双语模型卡中的在线测试入口，不重新上传 GGUF 权重。"""

from __future__ import annotations

import os
from pathlib import Path


# 三张远端卡共用两个 UTF-8 文档源文件，避免各平台文案漂移。
ROOT = Path(__file__).resolve().parents[1]
CARDS = {
    "README.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
    "README.zh-CN.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
    "README.en.md": ROOT / "docs" / "HF_MODEL_CARD.md",
}


def main() -> None:
    """按顺序更新 Hugging Face 和 ModelScope 的三张卡。"""
    from huggingface_hub import CommitOperationAdd, HfApi
    from modelscope_hub import HubApi

    # Hugging Face 凭据从本机 Hub 登录缓存读取，不进入仓库源码。
    hf = HfApi()
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                  for name, path in CARDS.items()]
    commit = hf.create_commit(repo_id="jinghao1632/bit-jev-2b-distilled",
                              repo_type="model", operations=operations,
                              commit_message="Add bilingual live demo links to model cards")
    print("Hugging Face model cards:", commit.commit_url)
    # ModelScope 令牌同样只从调用进程的环境变量读取。
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError("缺少 MODELSCOPE_API_TOKEN；Hugging Face 卡已更新")
    ms = HubApi(token=token)
    for name, path in CARDS.items():
        ms.upload_file("JingHao9616/bit-jev-2b-distilled", "model", path, name,
                       commit_message="Add bilingual live demo links to model cards")
    print("ModelScope model cards: 3 files")


if __name__ == "__main__":
    main()
