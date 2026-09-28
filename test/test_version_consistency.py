"""核对源码元数据与运行时版本，防止原生缓存被旧版本号复用。"""

import sys
import tomllib
import unittest
from pathlib import Path


# 测试入口固定使用项目 core 包，避免本机已安装的 PyPI 版本干扰。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))

import bit_jev  # noqa: E402
from bit_jev.native_build import _build_location  # noqa: E402


class VersionConsistencyTests(unittest.TestCase):
    """校验公开包版本、导入版本与原生缓存目录相同。"""

    def test_runtime_and_native_cache_match_metadata(self):
        """缓存必须按当前发行版本隔离，避免加载旧二进制。"""
        # Python 3.11 起的 tomllib 直接读取发行元数据，不执行打包脚本。
        project = tomllib.loads((PROJECT_ROOT / "core" / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["project"]["version"]
        self.assertEqual(bit_jev.__version__, version)
        _, build_dir, _ = _build_location("cpu", PROJECT_ROOT / "test")
        self.assertEqual(build_dir.parts[-2], version)


if __name__ == "__main__":
    unittest.main()
