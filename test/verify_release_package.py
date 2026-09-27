"""逐项校验 CPU 发布归档的安全路径、文件大小与 SHA-256。"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest_stream(stream):
    """分块读取归档成员，避免把大模型一次性装入内存。"""
    # 摘要对象记录当前归档成员的原始字节。
    digest = hashlib.sha256()
    # 字节计数用于核对清单中的文件大小。
    size = 0
    # 每轮最多读取 8 MiB，模型文件不会占满内存。
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify(archive_path):
    """流式读取所有成员，并与归档末尾的清单严格比对。"""
    # 顶层目录名由压缩包文件名确定，禁止嵌入其他目录。
    expected_root = archive_path.name.removesuffix(".tar.gz")
    if expected_root == archive_path.name:
        raise ValueError("发布包必须使用 .tar.gz 扩展名")
    # 逐项摘要先保存在内存；模型文件内容不会被保留。
    observed = {}
    # 清单正文在循环结束后解析，要求它位于唯一的顶层目录下。
    manifest_bytes = None
    with tarfile.open(archive_path, mode="r|gz") as archive:
        # 每个成员必须是普通文件，路径必须位于版本目录内。
        for member in archive:
            name = member.name
            parts = Path(name).parts
            if not member.isfile() or len(parts) != 2 or parts[0] != expected_root or parts[1] in {"", ".", ".."}:
                raise ValueError(f"归档成员路径或类型无效：{name}")
            # 同名文件会让解压结果不确定，因此直接拒绝。
            relative_name = parts[1]
            if relative_name in observed or (relative_name == "SHA256SUMS.json" and manifest_bytes is not None):
                raise ValueError(f"归档成员重复：{name}")
            # 所有成员都有输入流；异常成员不得被跳过。
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"归档成员不可读取：{name}")
            if relative_name == "SHA256SUMS.json":
                manifest_bytes = stream.read()
            else:
                observed[relative_name] = digest_stream(stream)
    if manifest_bytes is None:
        raise ValueError("缺少 SHA256SUMS.json")
    # 清单必须完整覆盖归档内容，不接受未列入清单的额外文件。
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("release") != expected_root or manifest.get("files") != observed:
        raise ValueError("清单与归档内文件的大小或 SHA-256 不一致")
    # 压缩包摘要供发布说明与下载后的外部校验使用。
    with archive_path.open("rb") as source:
        archive_hash = digest_stream(source)["sha256"]
    result = {"archive": str(archive_path), "sha256": archive_hash, "files_verified": len(observed)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    """解析待验证的压缩包路径并执行完整性检查。"""
    # 参数由调用者提供，文件必须已经存在。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verify(args.archive)


if __name__ == "__main__":
    main()
