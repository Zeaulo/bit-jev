"""将同一组公开推理文件和双语模型卡同步到两个模型站点。"""

import argparse
import json
import os
import shutil
from pathlib import Path


# 所有暂存与运行文件均保存在项目 test 目录；访问令牌只从环境读取。
ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "jinghao1632/bit-jev-2b-distilled"
MS_REPO = "JingHao9616/bit-jev-2b-distilled"
SOURCE_DIR = ROOT / "test" / "release" / "hf-bit-jev-2b-distilled"
STAGE_DIR = ROOT / "test" / "release" / "modelscope-bit-jev-2b-distilled"
MODEL_FILES = (
    "backbone-i2_s.gguf", "head.f32", "chat_template.jinja", "config.json",
    "pointer.json", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "benchmark_case_autodl.json",
)
FIGURES = (
    "bit-jev-logo.png", "project-flow.zh-CN.png", "project-flow.en.png",
    "bit-jev-case-speed.zh-CN.png", "bit-jev-case-speed.en.png",
    "bit-jev-case-memory.zh-CN.png", "bit-jev-case-memory.en.png",
)


def prepare_stage() -> Path:
    """复用已校验的大模型文件，并写入最新双语模型卡与图片。"""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    # 模型主体使用硬链接以免复制约 1.19 GB；跨卷时再复制。
    for filename in MODEL_FILES:
        source_path = SOURCE_DIR / filename
        target_path = STAGE_DIR / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not target_path.exists():
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
    # 中文为两个模型站点的默认模型卡，英文另存 README.en.md。
    card_files = {
        "README.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
        "README.zh-CN.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
        "README.en.md": ROOT / "docs" / "HF_MODEL_CARD.md",
    }
    for filename, source_path in card_files.items():
        shutil.copy2(source_path, STAGE_DIR / filename)
    # 模型卡引用的每张图都必须是仓库根目录中的真实文件。
    for filename in FIGURES:
        shutil.copy2(ROOT / "docs" / "figures" / filename, STAGE_DIR / filename)
    shutil.copy2(ROOT / "docs" / "benchmark-data" /
                 "bit-jev-epyc9654-cpu32-2026-09-28.json",
                 STAGE_DIR / "benchmark_case_epyc.json")
    # 两个站点记录同一组模型文件的哈希，仓库名按目标分别标注。
    manifest = json.loads((SOURCE_DIR / "SHA256SUMS.json").read_text(encoding="utf-8"))
    manifest["repository"] = MS_REPO
    (STAGE_DIR / "SHA256SUMS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 上传前验证每个引用图片和模型主体的大小，避免发布残缺目录。
    for filename in (*MODEL_FILES, *FIGURES, "README.md", "README.en.md",
                     "README.zh-CN.md", "SHA256SUMS.json", "benchmark_case_epyc.json"):
        if (STAGE_DIR / filename).stat().st_size <= 0:
            raise ValueError(f"暂存文件为空：{filename}")
    return STAGE_DIR


def publish_huggingface() -> None:
    """更新 Hugging Face 上的中文默认卡、英文卡和实际引用图片。"""
    from huggingface_hub import CommitOperationAdd, HfApi

    # 直接读取文档源文件，避免另一个站点上传大文件时改写暂存目录。
    publish_files = {
        "README.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
        "README.zh-CN.md": ROOT / "docs" / "HF_MODEL_CARD.zh-CN.md",
        "README.en.md": ROOT / "docs" / "HF_MODEL_CARD.md",
        "benchmark_case_epyc.json": ROOT / "docs" / "benchmark-data" /
                                    "bit-jev-epyc9654-cpu32-2026-09-28.json",
    }
    for filename in FIGURES:
        publish_files[filename] = ROOT / "docs" / "figures" / filename
    operations = [CommitOperationAdd(path_in_repo=filename,
                                     path_or_fileobj=source_path)
                  for filename, source_path in publish_files.items()]
    api = HfApi()
    api.create_commit(repo_id=HF_REPO, repo_type="model", operations=operations,
                      commit_message="Update bilingual model cards and benchmark figures")
    remote_files = set(api.list_repo_files(repo_id=HF_REPO, repo_type="model"))
    missing_files = set(publish_files) - remote_files
    if missing_files:
        raise RuntimeError(f"Hugging Face 提交后缺少文件：{sorted(missing_files)}")
    print(f"Hugging Face 已验证 {len(publish_files)} 个文档与图片文件")


def publish_modelscope() -> None:
    """在登录账号中创建公开模型仓库并上传完整 GGUF 推理包。"""
    from modelscope_hub import HubApi

    if not os.environ.get("MODELSCOPE_API_TOKEN"):
        raise RuntimeError("请通过 MODELSCOPE_API_TOKEN 环境变量提供发布令牌")
    stage_dir = prepare_stage()
    api = HubApi()
    username = api.whoami().username
    if username.casefold() != MS_REPO.split("/")[0].casefold():
        raise RuntimeError(f"ModelScope 登录账号 {username} 与目标命名空间不一致")
    if not api.repo_exists(MS_REPO, "model"):
        api.create_repo(MS_REPO, "model", visibility="public",
                        chinese_name="bit-jev 结构化决策模型",
                        description="BitNet I2_S GGUF structured decision model")
    api.upload_folder(MS_REPO, "model", stage_dir,
                      commit_message="Publish bit-jev GGUF model and bilingual model cards",
                      max_workers=4)
    # 远端文件清单至少应含权重、指针头、默认模型卡和全部图片。
    remote_files = {item.path for item in api.list_repo_files(MS_REPO, "model")}
    required_files = {"backbone-i2_s.gguf", "head.f32", "README.md",
                      "README.en.md", "SHA256SUMS.json", *FIGURES}
    missing_files = required_files - remote_files
    if missing_files:
        raise RuntimeError(f"ModelScope 上传后缺少文件：{sorted(missing_files)}")
    print(f"ModelScope 已验证 {len(required_files)} 个关键文件")


def main() -> None:
    """通过参数选择只准备文件或发布到指定站点。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", choices=("prepare", "huggingface", "modelscope"),
                        required=True)
    args = parser.parse_args()
    if args.hub == "prepare":
        print(prepare_stage())
    elif args.hub == "huggingface":
        publish_huggingface()
    else:
        publish_modelscope()


if __name__ == "__main__":
    main()
