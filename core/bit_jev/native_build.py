"""从固定上游源码按需构建 bit-jev GGUF 原生推理程序。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import __version__


# 固定提交与仓库 test/bootstrap_bitnet.py 使用的已验证源码相同。
BITNET_COMMIT = "0b341e582afbf9e1011f24744b554c96a3477eb5"
LLAMA_COMMIT = "390c307752ab78fd8189f359d6954c9ba1be74af"
BITNET_URL = "https://github.com/microsoft/BitNet.git"


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


def build_native(device: str = "cpu", *, cache_dir: str | Path | None = None,
                 source_dir: str | Path | None = None, jobs: int = 4) -> Path:
    """返回原生程序路径；首次使用时准备依赖并按设备构建。"""
    # gpu 别名选择跨平台 Vulkan；需要 NVIDIA CUDA 内核时显式指定 cuda。
    backend = "vulkan" if device == "gpu" else device
    if backend not in {"cpu", "vulkan", "cuda"}:
        raise ValueError("device 必须为 cpu、gpu、vulkan 或 cuda")
    if jobs < 1:
        raise ValueError("jobs 必须大于零")
    # 把构建缓存放在用户目录，pip wheel 和工作区都不会写入依赖源码。
    cache = Path(cache_dir) if cache_dir else Path(os.environ.get("BIT_JEV_CACHE", Path.home() / ".cache" / "bit-jev"))
    cache = cache.expanduser().resolve()
    binary_name = "bit-jev-cpu.exe" if os.name == "nt" else "bit-jev-cpu"
    build_dir = cache / "build" / __version__ / backend
    binary = build_dir / "bin" / binary_name
    if not binary.is_file():
        binary = build_dir / binary_name
    if binary.is_file():
        return binary
    if not shutil.which("cmake") or not shutil.which("git"):
        raise RuntimeError("首次构建需要 Git、CMake 3.28+ 和 C++17 编译器")
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
    _run(options)
    _run(["cmake", "--build", str(build_dir), "--target", "bit-jev-cpu", "--parallel", str(jobs)])
    for candidate in (build_dir / "bin" / binary_name, build_dir / binary_name,
                      build_dir / "Release" / binary_name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("CMake 报告构建成功，但未找到 bit-jev 原生程序")
