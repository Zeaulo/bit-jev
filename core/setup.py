"""构建 wheel 时附加源码和 Windows x64 CPU 预编译程序。"""

import platform
import sys
from pathlib import Path
from shutil import copy2

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def _supports_windows_binary():
    """只在 64 位 x86 Windows 上发放对应架构的 CPU 程序。"""
    # 可执行文件使用 AVX2 指令，但不与 CPython 的 ABI 绑定。
    return sys.platform == "win32" and platform.machine().lower() in {"amd64", "x86_64"} and sys.maxsize > 2**32


class BuildWithNative(build_py):
    """在标准 Python 构建完成后附加原生文件。"""

    def run(self):
        """复制源码、补丁及与目标平台匹配的预编译程序。"""
        super().run()
        # 构建目录由 setuptools 提供，源目录相对本 setup.py 固定。
        native_source = Path(__file__).resolve().parent / "native"
        native_target = Path(self.build_lib) / "bit_jev" / "_native"
        native_target.mkdir(parents=True, exist_ok=True)
        # 三份文件是按需构建的最小输入，不把约 1.19 GB 的权重放进 wheel。
        for filename in ("main.cpp", "CMakeLists.txt", "llama-relu2.patch"):
            copy2(native_source / filename, native_target / filename)
        # 只有 Windows x64 wheel 带可执行文件；其他平台保留源码构建路径。
        if _supports_windows_binary():
            binary_target = Path(self.build_lib) / "bit_jev" / "_bin"
            binary_target.mkdir(parents=True, exist_ok=True)
            copy2(native_source / "prebuilt" / "win_amd64" / "bit-jev-cpu.exe",
                  binary_target / "bit-jev-cpu.exe")
            # 第三方 MIT 许可文本随二进制一起安装，满足再分发要求。
            for filename in ("BITNET_LICENSE", "LLAMA_CPP_LICENSE"):
                copy2(native_source / "prebuilt" / "licenses" / filename,
                      binary_target / filename)


class PlatformWheel(bdist_wheel):
    """把含 Windows 可执行文件的 wheel 标成特定平台。"""

    def finalize_options(self):
        """Windows 发行包使用平台布局，源码型发行包保持通用布局。"""
        super().finalize_options()
        if _supports_windows_binary():
            self.root_is_pure = False
            self.plat_name = "win_amd64"

    def get_tag(self):
        """原生程序不链接 Python，故可被 3.11 和 3.12 共用。"""
        if _supports_windows_binary():
            return "py3", "none", "win_amd64"
        return super().get_tag()


class PlatformDistribution(Distribution):
    """让 setuptools 把携带可执行文件的包安装到平台库目录。"""

    def has_ext_modules(self):
        """Windows x64 二进制虽非 Python 扩展，仍属于平台专用产物。"""
        return _supports_windows_binary()


setup(cmdclass={"build_py": BuildWithNative, "bdist_wheel": PlatformWheel},
      distclass=PlatformDistribution)
