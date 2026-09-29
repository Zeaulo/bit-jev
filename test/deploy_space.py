"""将仓库内的在线演示源码发布到指定模型社区空间。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


# 仅传输公开演示源码；令牌只从环境或本机 Hub 登录缓存读取。
SPACE_DIR = Path(__file__).resolve().parents[1] / "spaces" / "bit_jev_demo"
HF_STATIC_DIR = Path(__file__).resolve().parents[1] / "spaces" / "hf_static"
MODELSCOPE_REPO = "JingHao9616/bit-jev-demo"
HUGGINGFACE_REPO = "jinghao1632/bit-jev-demo"


def deploy_modelscope() -> None:
    """上传 ModelScope Studio 文件并请求免费资源重新部署。"""
    from modelscope_hub import HubApi

    # SDK 支持 studio 类型上传；CLI upload 子命令目前只开放 model/dataset。
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError("请通过 MODELSCOPE_API_TOKEN 环境变量提供发布令牌")
    api = HubApi(token=token)
    result = api.upload_folder(MODELSCOPE_REPO, "studio", SPACE_DIR,
                               path_in_repo="", commit_message="Update bilingual Choice Noul Score demo",
                               ignore_patterns=["__pycache__/*", "*.pyc", ".ms_upload_cache"],
                               disable_tqdm=True)
    print("ModelScope upload:", result)
    print("ModelScope deploy:", api.deploy_repo(MODELSCOPE_REPO, "studio"))


def deploy_huggingface() -> None:
    """上传免费静态入口；推理链接指向 ModelScope CPU 空间。"""
    from huggingface_hub import HfApi

    # 当前账号创建 Docker 计算 Space 返回 402；保持已创建的静态 Space 类型。
    api = HfApi()
    result = api.upload_folder(repo_id=HUGGINGFACE_REPO, repo_type="space",
                               folder_path=HF_STATIC_DIR, commit_message="Update three-mode live demo entry",
                               ignore_patterns=["__pycache__/*", "*.pyc"])
    print("Hugging Face upload:", result.commit_url)


def main() -> None:
    """从命令行选择一个发布站点，避免误改另一个站点。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("modelscope", "huggingface"))
    args = parser.parse_args()
    if args.target == "modelscope":
        deploy_modelscope()
    else:
        deploy_huggingface()


if __name__ == "__main__":
    main()
