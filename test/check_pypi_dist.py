"""发布前核对平台 wheel 与源码包包含原生程序且不携带模型权重。"""

import tarfile
import zipfile
from pathlib import Path


# PyPI 包只允许代码、许可证和小型原生构建输入。
DIST = Path(__file__).resolve().parent / "pypi-dist"
VERSION = "0.11.10"
WHEEL = DIST / f"bit_jev-{VERSION}-py3-none-win_amd64.whl"
SOURCE = DIST / f"bit_jev-{VERSION}.tar.gz"
REQUIRED_WHEEL = {"bit_jev/gguf.py", "bit_jev/demo.py", "bit_jev/encoding.py", "bit_jev/native_build.py",
                  "bit_jev/_native/main.cpp", "bit_jev/_native/CMakeLists.txt",
                  "bit_jev/_native/llama-relu2.patch", "bit_jev/_bin/bit-jev-cpu.exe",
                  "bit_jev/_bin/bit-jev-vulkan.exe",
                  "bit_jev/_bin/BITNET_LICENSE", "bit_jev/_bin/LLAMA_CPP_LICENSE"}
def main():
    """检查归档存在、文件清单完整，并阻止大文件误上传。"""
    with zipfile.ZipFile(WHEEL) as archive:
        wheel_names = set(archive.namelist())
        # 安装即用命令必须写进发行元数据，不能只在源码里存在。
        entry_file = f"bit_jev-{VERSION}.dist-info/entry_points.txt"
        entry_points = archive.read(entry_file).decode("utf-8") if entry_file in wheel_names else ""
    with tarfile.open(SOURCE, "r:gz") as archive:
        source_names = {name.removeprefix(f"bit_jev-{VERSION}/") for name in archive.getnames()}
    missing_wheel = REQUIRED_WHEEL - wheel_names
    missing_source = {"native/main.cpp", "native/CMakeLists.txt", "native/llama-relu2.patch",
                      "native/prebuilt/win_amd64/bit-jev-cpu.exe",
                      "native/prebuilt/win_amd64/bit-jev-vulkan.exe",
                      "native/prebuilt/licenses/BITNET_LICENSE",
                      "native/prebuilt/licenses/LLAMA_CPP_LICENSE",
                      "LICENSE", "setup.py"} - source_names
    if missing_wheel or missing_source:
        raise AssertionError(f"发行包文件缺失：wheel={missing_wheel}，sdist={missing_source}")
    if "bit-jev-demo = bit_jev.demo:main" not in entry_points:
        raise AssertionError("wheel 缺少安装即用命令 bit-jev-demo")
    if any(name.endswith((".gguf", ".safetensors", ".pt")) for name in wheel_names | source_names):
        raise AssertionError("发行包意外包含模型权重")
    if WHEEL.stat().st_size > 50_000_000 or SOURCE.stat().st_size > 50_000_000:
        raise AssertionError("发行包超过预期的 50 MB 上限")
    print(f"发行包检查通过：wheel {WHEEL.stat().st_size} 字节，sdist {SOURCE.stat().st_size} 字节")


if __name__ == "__main__":
    main()
