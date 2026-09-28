"""验证公开 GGUF 在不同模型站点之间的下载分流与故障回退。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# 测试当前项目源码，避免本机旧版安装覆盖待发布代码。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from bit_jev.gguf import DEFAULT_MODEL, DEFAULT_MODELSCOPE_MODEL, download_model  # noqa: E402


class ModelSourceTests(unittest.TestCase):
    """覆盖本地路径、显式站点选择和 Hugging Face 故障回退。"""

    def test_local_directory_never_uses_network(self):
        """本地模型目录应在无网络条件下直接返回。"""
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as hf_download:
                self.assertEqual(download_model(directory), Path(directory).resolve())
                hf_download.assert_not_called()

    def test_huggingface_failure_falls_back_to_modelscope(self):
        """默认路径在 Hugging Face 超时后应使用已映射的 ModelScope 仓库。"""
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download", side_effect=TimeoutError("连接超时")) as hf_download:
                with patch("modelscope_hub.HubApi") as hub_class:
                    hub_class.return_value.download_repo.return_value = Path(directory)
                    self.assertEqual(download_model(), Path(directory).resolve())
                    self.assertEqual(hf_download.call_args.kwargs["etag_timeout"], 5)
                    hub_class.return_value.download_repo.assert_called_once()
                    self.assertEqual(hub_class.return_value.download_repo.call_args.args[:2],
                                     (DEFAULT_MODELSCOPE_MODEL, "model"))

    def test_explicit_modelscope_skips_huggingface(self):
        """指定 ModelScope 时不应请求 Hugging Face。"""
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as hf_download:
                with patch("modelscope_hub.HubApi") as hub_class:
                    hub_class.return_value.download_repo.return_value = Path(directory)
                    self.assertEqual(download_model(DEFAULT_MODEL, source="modelscope"),
                                     Path(directory).resolve())
                    hf_download.assert_not_called()

    def test_invalid_source_is_rejected(self):
        """错误站点名称应在执行网络请求前被拒绝。"""
        with self.assertRaisesRegex(ValueError, "source"):
            download_model(source="unknown")


if __name__ == "__main__":
    unittest.main()
