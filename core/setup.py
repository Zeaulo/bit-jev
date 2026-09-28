"""构建 wheel 时把唯一的原生源码复制到 Python 包中。"""

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithNative(build_py):
    """在标准 Python 构建完成后附加按需编译所需的原生文件。"""

    def run(self):
        """复制源码、CMake 配置和固定上游补丁到 wheel 内部。"""
        super().run()
        # 构建目录由 setuptools 提供，源目录相对本 setup.py 固定。
        native_source = Path(__file__).resolve().parent / "native"
        native_target = Path(self.build_lib) / "bit_jev" / "_native"
        native_target.mkdir(parents=True, exist_ok=True)
        # 三份文件是按需构建的最小输入，不把约 1.19 GB 的权重放进 wheel。
        for filename in ("main.cpp", "CMakeLists.txt", "llama-relu2.patch"):
            copy2(native_source / filename, native_target / filename)


setup(cmdclass={"build_py": BuildWithNative})
