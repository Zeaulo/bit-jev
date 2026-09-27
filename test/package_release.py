"""打包可独立下载的 CPU I2_S 模型、分词器、示例和来源说明。"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


# 所有发布输入都从项目根目录解析，避免工作目录影响产物内容。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 归档内文件名与本地来源逐项对应，不携带完整训练检查点。
RELEASE_FILES = {
    "backbone-i2_s.gguf": "export/bit-jev-2b-distilled/backbone-i2_s.gguf",
    "head.f32": "export/bit-jev-2b-distilled/head.f32",
    "pointer.json": "export/bit-jev-2b-distilled/pointer.json",
    "config.json": "runs/bit-jev-2b-distilled/config.json",
    "tokenizer.json": "runs/bit-jev-2b-distilled/tokenizer.json",
    "tokenizer_config.json": "runs/bit-jev-2b-distilled/tokenizer_config.json",
    "special_tokens_map.json": "runs/bit-jev-2b-distilled/special_tokens_map.json",
    "chat_template.jinja": "runs/bit-jev-2b-distilled/chat_template.jinja",
    "example.jsonl": "test/example_cpu_request.jsonl",
    "MODEL_CARD.md": "docs/MODEL_CARD.md",
    "LICENSE": "LICENSE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
}


def sha256_file(path):
    """分块计算文件摘要，避免将大模型整体读入内存。"""
    # 摘要对象保存按顺序读取的字节指纹。
    digest = hashlib.sha256()
    # 输入文件以二进制形式读取，确保不同平台的换行不改变摘要。
    with path.open("rb") as source:
        # 每轮仅读取 8 MiB，给发布过程留出固定内存上限。
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package(version, output_dir):
    """先核对所有输入，再生成带逐文件 SHA-256 清单的发布包。"""
    # 发行目录名同时用作压缩包名和归档内的唯一顶层目录。
    release_name = f"bit-jev-2b-i2_s-v{version}"
    # 输出目录位于 test 下，防止大型压缩包进入源码发布。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 每个文件的路径和摘要在打包前固定，缺失文件立即报错。
    source_paths = {}
    file_manifest = {}
    for archive_name, relative_source in RELEASE_FILES.items():
        # 来源路径由白名单常量定义，不能由外部参数注入。
        source_path = PROJECT_ROOT / relative_source
        if not source_path.is_file():
            raise FileNotFoundError(f"发布输入缺失：{source_path}")
        source_paths[archive_name] = source_path
        file_manifest[archive_name] = {
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
    # 清单仅记录可移植的归档内名称，不泄露本机绝对路径。
    manifest = {"release": release_name, "format": "bit-jev-cpu-i2_s-v1", "files": file_manifest}
    # 序列化结果保持 UTF-8 与稳定键序，方便独立校验。
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # 模型与附属文件共用一个归档，下载者不必手动配对权重与决策头。
    archive_path = output_dir / f"{release_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        # 逐项归档已通过存在性和摘要检查的文件。
        for archive_name, source_path in source_paths.items():
            archive.add(source_path, arcname=f"{release_name}/{archive_name}", recursive=False)
        # 内存中的 SHA256SUMS.json 使用相同的归档顶层目录。
        manifest_info = tarfile.TarInfo(f"{release_name}/SHA256SUMS.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = 0
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
    # 归档期间若来源文件变化，原先写入的摘要将失效；发布前立即拒绝该产物。
    for archive_name, source_path in source_paths.items():
        if source_path.stat().st_size != file_manifest[archive_name]["bytes"] or sha256_file(source_path) != file_manifest[archive_name]["sha256"]:
            raise RuntimeError(f"打包期间来源文件发生变化：{source_path}")
    # 顶层摘要供 GitHub Release 页面和下载后的完整性校验使用。
    archive_digest = sha256_file(archive_path)
    summary = {"archive": str(archive_path), "bytes": archive_path.stat().st_size,
               "sha256": archive_digest, "files": len(source_paths)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    """解析发行版本与输出目录并执行打包。"""
    # 命令行参数只控制版本和输出目录，文件白名单保持固定。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "test/release"))
    args = parser.parse_args()
    package(args.version, Path(args.out_dir))


if __name__ == "__main__":
    main()
