"""验证首次加载的构建提示会在下载模型前出现。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# 测试从项目根目录运行时只导入本仓库的 Python 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from bit_jev import __version__
from bit_jev.gguf import BitJev
from bit_jev.native_build import preflight_native


class NativePreflightTests(unittest.TestCase):
    """覆盖缺工具、旧版本和已缓存程序这三条首次加载分支。"""

    def test_missing_tools_report_links_before_download(self):
        """缺少 Git 和 CMake 时指出下载地址，且不触发模型下载。"""
        # 把工具检测固定为缺失，避免测试机已安装的工具影响结果。
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "test") as temporary_cache:
            with patch("bit_jev.native_build.shutil.which", return_value=None):
                with patch("bit_jev.gguf.download_model") as download:
                    with self.assertRaisesRegex(RuntimeError, "Git, CMake 3.28") as caught:
                        BitJev.from_pretrained(device="cpu", native_cache=temporary_cache)
                    download.assert_not_called()
        self.assertIn("https://git-scm.com/install/", str(caught.exception))
        self.assertIn("https://cmake.org/download/", str(caught.exception))
        self.assertIn("正常推理时不使用 Git", str(caught.exception))

    def test_old_cmake_reports_minimum_version(self):
        """CMake 版本低于 3.28 时提前给出下载地址。"""
        # Git 和 CMake 均可见，版本输出模拟用户机器上的旧版工具。
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "test") as temporary_cache:
            with patch("bit_jev.native_build.shutil.which", side_effect=lambda name: name):
                with patch("bit_jev.native_build._run", return_value="cmake version 3.27.9"):
                    with self.assertRaisesRegex(RuntimeError, "CMake 3.28") as caught:
                        preflight_native(cache_dir=temporary_cache)
        self.assertIn("https://cmake.org/download/", str(caught.exception))

    def test_existing_binary_skips_build_tools(self):
        """已有对应版本的原生程序时不要求再次安装构建工具。"""
        # 模拟 CMake 多配置生成器的 Release 产物位置。
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "test") as temporary_cache:
            binary_name = "bit-jev-cpu.exe" if sys.platform == "win32" else "bit-jev-cpu"
            binary = Path(temporary_cache) / "build" / __version__ / "cpu" / "Release" / binary_name
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch("bit_jev.native_build.shutil.which", return_value=None):
                self.assertEqual(preflight_native(cache_dir=temporary_cache), binary)

    def test_local_source_does_not_require_git(self):
        """调用方已准备原生源码时只要求 CMake 构建工具。"""
        # 本地源码的获取与补丁责任由调用方承担，不需要包再次执行 Git。
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "test") as temporary_cache:
            with patch("bit_jev.native_build.shutil.which",
                       side_effect=lambda name: None if name == "git" else "cmake"):
                with patch("bit_jev.native_build._run", return_value="cmake version 3.28.0"):
                    self.assertIsNone(preflight_native(
                        cache_dir=temporary_cache, source_dir=PROJECT_ROOT / "learning"))

    def test_supplied_binary_skips_preflight(self):
        """调用方提供已构建程序时不检查 Git 或 CMake。"""
        # 构造函数和下载器在此仅作为隔离点，不执行真实网络或原生程序。
        with patch("bit_jev.gguf.preflight_native") as preflight:
            with patch("bit_jev.gguf.download_model", return_value=PROJECT_ROOT):
                with patch.object(BitJev, "__init__", return_value=None):
                    BitJev.from_pretrained(binary="existing-runner")
        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
