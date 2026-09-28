"""从固定上游源码按需构建 bit-jev GGUF 原生推理程序。"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path

from . import __version__


# 固定提交与仓库 test/bootstrap_bitnet.py 使用的已验证源码相同。
BITNET_COMMIT = "0b341e582afbf9e1011f24744b554c96a3477eb5"
LLAMA_COMMIT = "390c307752ab78fd8189f359d6954c9ba1be74af"
BITNET_URL = "https://github.com/microsoft/BitNet.git"
GIT_INSTALL_URL = "https://git-scm.com/install/"
CMAKE_INSTALL_URL = "https://cmake.org/download/"
WINDOWS_CPP_INSTALL_URL = "https://learn.microsoft.com/cpp/build/vscpp-step-0-installation"


def _run(arguments: list[str], *, directory: Path | None = None) -> str:
    """用参数数组执行外部程序，并保留可读的失败诊断。"""
    # 用户路径不会进入 shell，从而避免路径中的空格或元字符改变命令含义。
    result = subprocess.run(arguments, cwd=directory, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()[-4000:]
        raise RuntimeError(f"原生构建命令失败：{' '.join(arguments)}\n{detail}")
    return result.stdout.strip()


def _native_source() -> Path:
    """优先读取 wheel 内文件；editable 安装回退到同仓库唯一的原生源码。"""
    # setuptools 在 wheel 构建时将 core/native 的文件复制到此目录。
    package_copy = Path(__file__).resolve().parent / "_native"
    if (package_copy / "CMakeLists.txt").is_file():
        return package_copy
    editable_source = Path(__file__).resolve().parents[1] / "native"
    if (editable_source / "CMakeLists.txt").is_file():
        return editable_source
    raise FileNotFoundError("bit-jev 包中缺少原生构建文件")


def _prepare_upstream(cache: Path) -> Path:
    """下载固定版本的 BitNet 与 llama.cpp，并仅应用运行时所需补丁。"""
    # 已完成的缓存仍核对两个提交，防止误用被替换的依赖目录。
    cache.mkdir(parents=True, exist_ok=True)
    bitnet = cache / "bitnet"
    if not bitnet.is_dir():
        _run(["git", "clone", "--filter=blob:none", BITNET_URL, str(bitnet)])
        _run(["git", "checkout", "--detach", BITNET_COMMIT], directory=bitnet)
    if _run(["git", "rev-parse", "HEAD"], directory=bitnet) != BITNET_COMMIT:
        raise RuntimeError("BitNet 缓存提交与包要求的固定提交不一致")
    llama = bitnet / "3rdparty" / "llama.cpp"
    if not (llama / ".git").exists():
        _run(["git", "submodule", "update", "--init", "--depth", "1", "--", "3rdparty/llama.cpp"],
             directory=bitnet)
    if _run(["git", "rev-parse", "HEAD"], directory=llama) != LLAMA_COMMIT:
        raise RuntimeError("llama.cpp 缓存提交与包要求的固定提交不一致")
    patch = _native_source() / "llama-relu2.patch"
    # 补丁只更换蒸馏检查点需要的 ReLU²，不改动用户自己的源码目录。
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=llama,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if check.returncode:
        _run(["git", "apply", "--check", str(patch)], directory=llama)
        _run(["git", "apply", str(patch)], directory=llama)
    return llama


def _build_location(device: str, cache_dir: str | Path | None) -> tuple[str, Path, str]:
    """计算设备对应的构建目录和程序名，并校验设备参数。"""
    # gpu 是 Vulkan 的便捷别名，缓存按实际后端分开保存。
    backend = "vulkan" if device == "gpu" else device
    if backend not in {"cpu", "vulkan", "cuda"}:
        raise ValueError("device 必须为 cpu、gpu、vulkan 或 cuda")
    # 用户可以通过环境变量或参数将原生缓存放到其他磁盘。
    cache = Path(cache_dir) if cache_dir else Path(
        os.environ.get("BIT_JEV_CACHE", Path.home() / ".cache" / "bit-jev"))
    build_dir = cache.expanduser().resolve() / "build" / __version__ / backend
    binary_name = "bit-jev-cpu.exe" if os.name == "nt" else "bit-jev-cpu"
    return backend, build_dir, binary_name


def _existing_binary(build_dir: Path, binary_name: str) -> Path | None:
    """复用不同 CMake 生成器已经产出的原生程序。"""
    # 单配置和多配置生成器会把程序放在不同的目录。
    for candidate in (build_dir / "bin" / binary_name, build_dir / binary_name,
                      build_dir / "Release" / binary_name):
        if candidate.is_file():
            return candidate
    return None


def _bundled_binary(device: str, source_dir: str | Path | None) -> Path | None:
    """寻找适用于当前 Windows x64 CPU 的 wheel 内预编译程序。"""
    # 指定本地原生源码意味着调用方明确要求重新构建，不能被预编译程序覆盖。
    if device != "cpu" or source_dir is not None or sys.platform != "win32":
        return None
    if platform.machine().lower() not in {"amd64", "x86_64"} or sys.maxsize <= 2**32:
        return None
    binary = Path(__file__).resolve().parent / "_bin" / "bit-jev-cpu.exe"
    if not binary.is_file():
        return None
    # 当前 I2_S 内核需要 AVX2；提前检查可避免 Windows 在启动时直接报非法指令。
    avx2_available = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    if not avx2_available:
        raise RuntimeError("当前 Windows x64 CPU 不支持 AVX2，无法运行 bit-jev 预编译程序。"
                           "请使用支持 AVX2 的机器，或传入适配本机的 binary 参数。")
    return binary


def preflight_native(device: str = "cpu", *, cache_dir: str | Path | None = None,
                     source_dir: str | Path | None = None) -> Path | None:
    """在下载大模型前检查首次原生构建需要的工具。"""
    # 已有编译产物不再依赖 Git、CMake 或本机 C++ 工具链。
    _, build_dir, binary_name = _build_location(device, cache_dir)
    cached_binary = _existing_binary(build_dir, binary_name)
    if cached_binary is not None:
        return cached_binary
    packaged_binary = _bundled_binary(device, source_dir)
    if packaged_binary is not None:
        return packaged_binary
    # 只有默认自动获取固定上游源码时才需要 Git；可信本地源码跳过检出。
    missing = []
    if source_dir is None and not shutil.which("git"):
        missing.append("Git")
    cmake = shutil.which("cmake")
    if not cmake:
        missing.append("CMake 3.28+")
    if missing:
        raise RuntimeError(
            f"首次加载需要编译原生推理程序，当前缺少：{', '.join(missing)}。\n"
            "Git 用于获取固定版本的 BitNet 与 llama.cpp 源码、核对提交并应用 ReLU² 补丁；"
            "正常推理时不使用 Git。\n"
            f"Git 下载：{GIT_INSTALL_URL}\n"
            f"CMake 下载：{CMAKE_INSTALL_URL}\n"
            f"Windows C++ 编译工具：{WINDOWS_CPP_INSTALL_URL}\n"
            "安装后重新打开终端；若已有自行编译的原生程序，可传入 binary 参数跳过自动构建。")
    # CMake 过旧时提前提示，避免下载约 1.19 GB 模型之后才由配置步骤报错。
    version_output = _run([cmake, "--version"])
    match = re.search(r"cmake version (\d+)\.(\d+)", version_output)
    if match is None or tuple(map(int, match.groups())) < (3, 28):
        raise RuntimeError(
            f"首次构建需要 CMake 3.28+；当前版本输出：{version_output[:120]}。"
            f"下载：{CMAKE_INSTALL_URL}")
    return None


def build_native(device: str = "cpu", *, cache_dir: str | Path | None = None,
                 source_dir: str | Path | None = None, jobs: int = 4) -> Path:
    """返回原生程序路径；首次使用时准备依赖并按设备构建。"""
    if jobs < 1:
        raise ValueError("jobs 必须大于零")
    # 预检查与模型下载前的检查共用一套错误信息。
    cached_binary = preflight_native(device, cache_dir=cache_dir, source_dir=source_dir)
    if cached_binary is not None:
        return cached_binary
    # 把构建缓存放在用户目录，pip wheel 和工作区都不会写入依赖源码。
    backend, build_dir, binary_name = _build_location(device, cache_dir)
    cache = build_dir.parents[2]
    # 源码参数用于可信的本地 checkout；默认从固定公开提交下载。
    llama = Path(source_dir).resolve() if source_dir else _prepare_upstream(cache)
    if not (llama / "include" / "llama.h").is_file():
        raise FileNotFoundError(f"找不到 llama.cpp 源码：{llama}")
    options = ["cmake", "-S", str(_native_source()), "-B", str(build_dir),
               "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
               f"-DBITNET_CPP_DIR={llama}",
               f"-DGGML_VULKAN={'ON' if backend == 'vulkan' else 'OFF'}",
               f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}"]
    # Windows 的 MinGW 安装优先配合本机 g++；没有时交给 CMake 选默认生成器。
    if os.name == "nt" and shutil.which("mingw32-make"):
        options.extend(["-G", "MinGW Makefiles"])
    if backend == "vulkan" and os.name == "nt" and shutil.which("glslc"):
        # MSYS2 Vulkan SDK 常不设置 VULKAN_SDK；由 glslc 推导同一前缀的头文件和库。
        sdk_root = Path(shutil.which("glslc")).resolve().parents[1]
        vulkan_library = sdk_root / "lib" / "libvulkan-1.dll.a"
        if (sdk_root / "include" / "vulkan" / "vulkan.h").is_file() and vulkan_library.is_file():
            options.extend([f"-DVulkan_INCLUDE_DIR={sdk_root / 'include'}",
                            f"-DVulkan_LIBRARY={vulkan_library}",
                            f"-DVulkan_GLSLC_EXECUTABLE={sdk_root / 'bin' / 'glslc.exe'}"])
    try:
        _run(options)
    except RuntimeError as error:
        # Windows 的 CMake 配置错误常来自尚未安装 C++ 工具链。
        if os.name == "nt":
            raise RuntimeError(f"{error}\nWindows C++ 编译工具：{WINDOWS_CPP_INSTALL_URL}") from error
        raise
    _run(["cmake", "--build", str(build_dir), "--target", "bit-jev-cpu", "--parallel", str(jobs)])
    compiled_binary = _existing_binary(build_dir, binary_name)
    if compiled_binary is not None:
        return compiled_binary
    raise FileNotFoundError("CMake 报告构建成功，但未找到 bit-jev 原生程序")
