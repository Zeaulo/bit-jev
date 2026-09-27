"""规范 Hugging Face 模型配置，并同步维护发布包的 SHA-256 清单。"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def normalize_config_bytes(config_path):
    """将 BitNet 配置中的 modules_to_not_convert 规范为数组。"""
    # 配置必须是 UTF-8 JSON 对象，避免悄悄覆盖未知格式。
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("模型 config.json 的顶层值必须是对象")

    # 只有存在量化配置时才检查 BitNet 量化排除名单。
    quantization = config.get("quantization_config")
    if isinstance(quantization, dict):
        excluded_modules = quantization.get("modules_to_not_convert")
        if excluded_modules is None:
            quantization["modules_to_not_convert"] = []
        elif not isinstance(excluded_modules, list):
            raise ValueError("quantization_config.modules_to_not_convert 必须是数组")

    # 稳定缩进和 UTF-8 编码，确保清单哈希可重复计算。
    return (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(content):
    """计算内存中字节内容的 SHA-256。"""
    # 哈希只作用于小型 JSON 元数据，不会读取模型权重到内存。
    return hashlib.sha256(content).hexdigest()


def repair_hf_package(package_dir):
    """修正现有 Hub 包配置，并更新清单中的配置大小与 SHA-256。"""
    # 所有操作限制在用户明确传入的包目录内。
    package_dir = Path(package_dir).resolve()
    config_path = package_dir / "config.json"
    manifest_path = package_dir / "SHA256SUMS.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("模型包必须同时包含 config.json 与 SHA256SUMS.json")

    # 先解析清单并确认其中恰有一条 config.json 记录。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("SHA256SUMS.json 的 files 字段必须是数组")
    config_records = [record for record in records if record.get("file") == "config.json"]
    if len(config_records) != 1:
        raise ValueError("SHA256SUMS.json 必须恰好包含一条 config.json 记录")

    # 先生成规范化内容，再写入临时文件以避免中断时留下半份 JSON。
    normalized_config = normalize_config_bytes(config_path)
    temporary_config = config_path.with_name("config.json.tmp")
    temporary_config.write_bytes(normalized_config)
    temporary_config.replace(config_path)

    # 清单时间采用 UTC，配置记录按已写入的实际字节重新计算。
    config_records[0]["bytes"] = len(normalized_config)
    config_records[0]["sha256"] = sha256_bytes(normalized_config)
    if "generated_utc" in manifest:
        manifest["generated_utc"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    normalized_manifest = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary_manifest = manifest_path.with_name("SHA256SUMS.json.tmp")
    temporary_manifest.write_bytes(normalized_manifest)
    temporary_manifest.replace(manifest_path)

    # 输出仅包含本次修改的元数据摘要，便于发布记录复核。
    result = {
        "package": str(package_dir),
        "config_bytes": len(normalized_config),
        "config_sha256": config_records[0]["sha256"],
        "modules_to_not_convert": json.loads(normalized_config)["quantization_config"]["modules_to_not_convert"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    """解析 Hub 模型包目录并执行配置与清单修复。"""
    # 命令行只接受一个包目录，默认为当前公开检查点暂存目录。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "release/hf-bit-jev-2b-distilled",
        help="包含 config.json 和 SHA256SUMS.json 的模型包目录",
    )
    args = parser.parse_args()
    repair_hf_package(args.package_dir)


if __name__ == "__main__":
    main()
