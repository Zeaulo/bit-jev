"""按固定提交和补丁准备 BitNet 原生依赖，并可选构建 CPU 程序。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


# 这些提交与当前训练导出、I2_S 推理及本地对照实验使用的源码一致。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BITNET_DIR = PROJECT_ROOT / "learning" / "bitnet"
LLAMA_DIR = BITNET_DIR / "3rdparty" / "llama.cpp"
PATCH_DIR = Path(__file__).resolve().parent / "patches"
BITNET_URL = "https://github.com/microsoft/bitnet.git"
BITNET_COMMIT = "0b341e582afbf9e1011f24744b554c96a3477eb5"
LLAMA_COMMIT = "390c307752ab78fd8189f359d6954c9ba1be74af"
BITNET_PATCH = PATCH_DIR / "bitnet-converter.patch"
LLAMA_PATCH = PATCH_DIR / "llama-relu2.patch"


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    """执行命令，并把失败转换成包含命令及输出的诊断。"""
    # 所有命令都以参数列表传递，路径和用户指定的生成器不会经过 Shell 解析。
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"命令失败（退出码 {result.returncode}）：{' '.join(command)}\n{output}")
    return (result.stdout or "").strip()


def git(repo: Path, *arguments: str, capture: bool = True) -> str:
    """在指定仓库执行 Git 命令。"""
    return run(["git", "-C", str(repo), *arguments], capture=capture)


def patch_applied(repo: Path, patch: Path) -> bool:
    """检查预期补丁是否已经应用，不修改已有工作区。"""
    result = subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", "--ignore-space-change", str(patch)],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


def ensure_patch(repo: Path, patch: Path, *, check_only: bool) -> None:
    """仅在补丁尚未应用且允许写入时应用一次。"""
    if patch_applied(repo, patch):
        print(f"已应用补丁：{patch.name}")
        return
    if check_only:
        raise RuntimeError(f"缺少预期补丁：{patch.name}")
    git(repo, "apply", "--check", "--ignore-space-change", str(patch))
    git(repo, "apply", "--ignore-space-change", str(patch), capture=False)
    if not patch_applied(repo, patch):
        raise RuntimeError(f"补丁应用后未通过反向检查：{patch.name}")
    print(f"已应用补丁：{patch.name}")


def verify_commit(repo: Path, expected: str) -> None:
    """阻止以其他上游版本构建未经验证的模型图。"""
    actual = git(repo, "rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(f"源码版本不符：{repo}\n当前 {actual}\n期望 {expected}")


def verify_exact_patch(repo: Path, relative_file: str, patch: Path) -> None:
    """比较补丁正文与当前差异，防止目标文件夹带未公布的额外修改。"""
    # 两份文件可能有不同的行尾；Git 的选项只忽略行尾空白，不忽略代码内容。
    current_diff = git(repo, "diff", "--no-ext-diff", "--no-color", "--unified=3", "--ignore-space-at-eol", "--", relative_file)
    expected_diff = patch.read_text(encoding="utf-8")

    def hunks(diff: str) -> str:
        """去掉可变的索引元数据，仅保留完整的补丁块。"""
        lines = diff.splitlines()
        first = next((index for index, line in enumerate(lines) if line.startswith("@@ ")), None)
        return "\n".join(lines[first:]) if first is not None else ""

    if hunks(current_diff) != hunks(expected_diff):
        raise RuntimeError(f"当前文件与发布补丁不完全相同：{repo / relative_file}")


def verify_changes() -> None:
    """确认受跟踪文件只包含公布的两处本地补丁及子模块状态。"""
    # 根仓库会把子模块内的补丁显示为一个修改项。
    allowed_bitnet = {"3rdparty/llama.cpp", "utils/convert-hf-to-gguf-bitnet.py"}
    allowed_llama = {"src/models/bitnet.cpp"}
    for repo, allowed in ((BITNET_DIR, allowed_bitnet), (LLAMA_DIR, allowed_llama)):
        # 暂存区应为空；工作区中的其他改动可能使复现实验失去意义。
        staged = git(repo, "diff", "--cached", "--name-only")
        if staged:
            raise RuntimeError(f"依赖仓库有暂存的改动：{repo}\n{staged}")
        changed = set(git(repo, "diff", "--name-only").splitlines())
        unexpected = changed - allowed
        if unexpected:
            raise RuntimeError(f"依赖仓库有未公布的改动：{repo}\n{sorted(unexpected)}")
    verify_exact_patch(BITNET_DIR, "utils/convert-hf-to-gguf-bitnet.py", BITNET_PATCH)
    verify_exact_patch(LLAMA_DIR, "src/models/bitnet.cpp", LLAMA_PATCH)


def prepare(*, check_only: bool) -> None:
    """克隆或校验固定版本的 BitNet，初始化子模块并应用补丁。"""
    for patch in (BITNET_PATCH, LLAMA_PATCH):
        if not patch.is_file():
            raise RuntimeError(f"缺少补丁文件：{patch}")

    if not BITNET_DIR.exists():
        if check_only:
            raise RuntimeError(f"源码目录不存在：{BITNET_DIR}")
        BITNET_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", BITNET_URL, str(BITNET_DIR)])
        git(BITNET_DIR, "checkout", "--detach", BITNET_COMMIT, capture=False)

    verify_commit(BITNET_DIR, BITNET_COMMIT)
    if not LLAMA_DIR.exists() or not (LLAMA_DIR / ".git").exists():
        if check_only:
            raise RuntimeError(f"llama.cpp 子模块尚未初始化：{LLAMA_DIR}")
        # 上游 llama.cpp 仓库历史较大；只获取父仓库固定的子模块提交。
        git(BITNET_DIR, "submodule", "update", "--init", "--depth", "1", "--", "3rdparty/llama.cpp", capture=False)

    verify_commit(LLAMA_DIR, LLAMA_COMMIT)
    ensure_patch(BITNET_DIR, BITNET_PATCH, check_only=check_only)
    ensure_patch(LLAMA_DIR, LLAMA_PATCH, check_only=check_only)
    verify_changes()
    print("BitNet、llama.cpp 的提交和补丁已核对。")


def build(*, generator: str | None, jobs: int, build_dir: Path) -> None:
    """按当前机器指令集配置并构建原生 CPU 推理程序。"""
    if shutil.which("cmake") is None:
        raise RuntimeError("未找到 CMake；构建需要 CMake 3.28+ 和 C++17 编译器。")
    # CMake 入口通过固定相对路径引用上述源码；不得把其他 checkout 当作构建输入。
    command = [
        "cmake", "-S", str(PROJECT_ROOT / "core" / "native"),
        "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
    ]
    if generator:
        command.extend(["-G", generator])
    run(command)
    run(["cmake", "--build", str(build_dir), "--target", "bit-jev-cpu", "--parallel", str(jobs)])
    print(f"原生 CPU 程序已构建：{build_dir}")


def main() -> None:
    """解析只读验证、依赖准备与可选构建参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查现有源码提交和补丁，不写入文件")
    parser.add_argument("--build", action="store_true", help="准备源码后构建原生 CPU 程序")
    parser.add_argument("--generator", help="可选 CMake 生成器；Windows MinGW 可传 MinGW Makefiles")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8), help="并行编译任务数")
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "core" / "build" / "bit-jev-cpu")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须大于零")
    if args.check and args.build:
        parser.error("--check 与 --build 不能同时使用")
    if shutil.which("git") is None:
        parser.error("未找到 Git")
    prepare(check_only=args.check)
    if args.build:
        build(generator=args.generator, jobs=args.jobs, build_dir=args.build_dir.resolve())


if __name__ == "__main__":
    main()
