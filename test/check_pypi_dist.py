"""发布前核对 wheel 与源码包包含原生构建文件且不携带模型权重。"""

import tarfile
import zipfile
from pathlib import Path


# PyPI 包只允许代码、许可证和小型原生构建输入。
DIST = Path(__file__).resolve().parent / "pypi-dist"
WHEEL = DIST / "bit_jev-0.8.7-py3-none-any.whl"
SOURCE = DIST / "bit_jev-0.8.7.tar.gz"
REQUIRED_WHEEL = {"bit_jev/gguf.py", "bit_jev/encoding.py", "bit_jev/native_build.py",
                  "bit_jev/_native/main.cpp", "bit_jev/_native/CMakeLists.txt",
                  "bit_jev/_native/llama-relu2.patch"}
def main():
    """检查归档存在、文件清单完整，并阻止大文件误上传。"""
    with zipfile.ZipFile(WHEEL) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(SOURCE, "r:gz") as archive:
        source_names = {name.removeprefix("bit_jev-0.8.7/") for name in archive.getnames()}
    missing_wheel = REQUIRED_WHEEL - wheel_names
    missing_source = {"native/main.cpp", "native/CMakeLists.txt", "native/llama-relu2.patch",
                      "LICENSE", "setup.py"} - source_names
    if missing_wheel or missing_source:
        raise AssertionError(f"发行包文件缺失：wheel={missing_wheel}，sdist={missing_source}")
    if any(name.endswith((".gguf", ".safetensors", ".pt")) for name in wheel_names | source_names):
        raise AssertionError("发行包意外包含模型权重")
    if WHEEL.stat().st_size > 1_000_000 or SOURCE.stat().st_size > 1_000_000:
        raise AssertionError("发行包超过预期的小型源码体积")
    print(f"发行包检查通过：wheel {WHEEL.stat().st_size} 字节，sdist {SOURCE.stat().st_size} 字节")


if __name__ == "__main__":
    main()
