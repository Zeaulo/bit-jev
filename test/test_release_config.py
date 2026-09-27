"""验证公开模型配置规范化与 Hugging Face 清单同步行为。"""

import hashlib
import json
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

import package_release
from release_config import normalize_config_bytes, repair_hf_package
from verify_release_package import verify


class ReleaseConfigTests(unittest.TestCase):
    """覆盖配置类型修复与发布哈希清单更新。"""

    def test_null_exclusion_list_becomes_empty_array(self):
        """Hugging Face 配置解析要求排除名单是数组。"""
        # 使用临时 JSON，避免测试修改真实发布包。
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({"quantization_config": {"modules_to_not_convert": None}}),
                encoding="utf-8",
            )
            normalized = json.loads(normalize_config_bytes(config_path))
        self.assertEqual(normalized["quantization_config"]["modules_to_not_convert"], [])

    def test_existing_array_is_preserved(self):
        """已经符合 schema 的模块名单不得被改写。"""
        # 保留配置提供者明确指定的模块列表。
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({"quantization_config": {"modules_to_not_convert": ["lm_head"]}}),
                encoding="utf-8",
            )
            normalized = json.loads(normalize_config_bytes(config_path))
        self.assertEqual(normalized["quantization_config"]["modules_to_not_convert"], ["lm_head"])

    def test_repair_updates_hub_manifest(self):
        """模型包修复后清单记录应匹配实际配置字节。"""
        # 临时目录模拟公开 Hub 包，不触碰检查点或网络服务。
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory)
            config_path = package_dir / "config.json"
            config_path.write_text(
                json.dumps({"quantization_config": {"modules_to_not_convert": None}}),
                encoding="utf-8",
            )
            manifest_path = package_dir / "SHA256SUMS.json"
            manifest_path.write_text(
                json.dumps({
                    "format": "bit-jev-sha256-v1",
                    "generated_utc": "2026-01-01T00:00:00Z",
                    "files": [{"file": "config.json", "bytes": 1, "sha256": "old"}],
                }),
                encoding="utf-8",
            )

            result = repair_hf_package(package_dir)
            config_bytes = config_path.read_bytes()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_record = manifest["files"][0]

        self.assertEqual(result["modules_to_not_convert"], [])
        self.assertEqual(config_record["bytes"], len(config_bytes))
        self.assertEqual(config_record["sha256"], hashlib.sha256(config_bytes).hexdigest())

    def test_tar_package_contains_normalized_config(self):
        """本地归档必须写入规范化配置并由逐项校验器接受。"""
        # 使用小型临时输入验证完整打包链，避免读取真实模型权重。
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            config_source = project_root / "runs/config.json"
            config_source.parent.mkdir(parents=True)
            config_source.write_text(
                json.dumps({"quantization_config": {"modules_to_not_convert": None}}),
                encoding="utf-8",
            )
            sample_source = project_root / "sample.txt"
            sample_source.write_text("sample", encoding="utf-8")
            release_files = {"config.json": "runs/config.json", "sample.txt": "sample.txt"}
            output_dir = project_root / "release"

            # 临时替换打包器的固定根目录和文件白名单，避免触碰真实检查点。
            with patch.object(package_release, "PROJECT_ROOT", project_root), patch.object(
                package_release, "RELEASE_FILES", release_files
            ):
                package_release.package("test", output_dir)

            archive_path = output_dir / "bit-jev-2b-i2_s-vtest.tar.gz"
            with tarfile.open(archive_path, "r:gz") as archive:
                config_member = archive.extractfile("bit-jev-2b-i2_s-vtest/config.json")
                manifest_member = archive.extractfile("bit-jev-2b-i2_s-vtest/SHA256SUMS.json")
                archived_config_bytes = config_member.read()
                archived_manifest = json.loads(manifest_member.read())
            archived_config = json.loads(archived_config_bytes)
            archived_record = archived_manifest["files"]["config.json"]

            # 既验证字段 schema，也保证归档清单对规范化字节的摘要准确。
            self.assertEqual(
                archived_config["quantization_config"]["modules_to_not_convert"], []
            )
            self.assertEqual(archived_record["bytes"], len(archived_config_bytes))
            self.assertEqual(
                archived_record["sha256"], hashlib.sha256(archived_config_bytes).hexdigest()
            )
            verify(archive_path)


if __name__ == "__main__":
    unittest.main()
