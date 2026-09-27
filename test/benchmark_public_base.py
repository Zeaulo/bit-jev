"""测量微软公开 BitNet 基础模型同一份 GGUF 在 CPU 和 GPU 上的原生吞吐。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


# 官方模型来源固定到已核对的提交；文件哈希仍须由本机重新计算。
OFFICIAL_REPOSITORY = "microsoft/bitnet-b1.58-2B-4T-gguf"
DEFAULT_REVISION = "9f43072f69492cbd5bc5d5ebb085fec519686a93"
OFFICIAL_MODEL_SHA256 = "13939ce5030319a35db346e5dba7a3a3bd599dfc18b113a2a97446ff964714c5"
OFFICIAL_MODEL_BYTES = 1_844_472_032
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test"
MIB = 1024 * 1024


def file_sha256(path: Path) -> str:
    """分块计算文件哈希，避免把整个 GGUF 读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * MIB):
            digest.update(block)
    return digest.hexdigest()


def git_value(directory: Path, *arguments: str) -> str | None:
    """读取仓库提交或状态；无 Git 信息时明确返回空值。"""
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_name() -> str:
    """尽可能取得可辨识的处理器型号。"""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as registry_key:
                return str(winreg.QueryValueEx(registry_key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or "unknown"


def gpu_rows(nvidia_smi: str) -> list[dict[str, Any]]:
    """读取 NVIDIA 设备型号、容量、驱动和当前全局显存。"""
    command = [
        nvidia_smi,
        "--query-gpu=index,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 5:
            continue
        try:
            rows.append({
                "index": int(fields[0].strip()),
                "name": fields[1].strip(),
                "memory_total_mib": float(fields[2].strip()),
                "memory_used_mib": float(fields[3].strip()),
                "driver_version": fields[4].strip(),
            })
        except ValueError:
            continue
    return rows


def gpu_process_used_mib(nvidia_smi: str, pid: int) -> float | None:
    """尝试读取指定进程显存；Windows WDDM/Vulkan 可能不提供此值。"""
    command = [
        nvidia_smi,
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 2:
            continue
        try:
            if int(fields[0].strip()) == pid:
                return float(fields[1].strip())
        except ValueError:
            continue
    return None


def gpu_by_index(nvidia_smi: str | None, index: int) -> dict[str, Any] | None:
    """在设备列表中取得指定 GPU 的当前快照。"""
    if nvidia_smi is None:
        return None
    return next((item for item in gpu_rows(nvidia_smi) if item["index"] == index), None)


def benchmark_rows(stdout: str, repetitions: int, prompt_tokens: int, decode_tokens: int) -> dict[str, Any]:
    """按 llama-bench 的 JSON samples_ns/samples_ts 字段校验并整理两种测试。"""
    try:
        raw_rows = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("llama-bench 未产生完整的 JSON 输出") from error
    if not isinstance(raw_rows, list) or len(raw_rows) != 2:
        raise ValueError("llama-bench 必须返回独立的 prefill 与 decode 两行")

    parsed: dict[str, Any] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("llama-bench 测试行必须是 JSON 对象")
        prompt_count = row.get("n_prompt")
        decode_count = row.get("n_gen")
        if (prompt_count, decode_count) == (prompt_tokens, 0):
            phase = "prefill"
            token_count = prompt_tokens
        elif (prompt_count, decode_count) == (0, decode_tokens):
            phase = "decode"
            token_count = decode_tokens
        else:
            raise ValueError(f"意外的 llama-bench 测试尺寸：{prompt_count}/{decode_count}")
        if phase in parsed:
            raise ValueError(f"llama-bench 重复返回了 {phase} 测试")
        nanoseconds = row.get("samples_ns")
        rates = row.get("samples_ts")
        if (
            not isinstance(nanoseconds, list)
            or not isinstance(rates, list)
            or len(nanoseconds) != repetitions
            or len(rates) != repetitions
        ):
            raise ValueError(f"{phase} 每次运行的样本数与 repetitions 不符")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in nanoseconds + rates):
            raise ValueError(f"{phase} 存在非法耗时或吞吐样本")
        # 用原生纳秒样本重新计算吞吐，避免只依赖汇总平均值。
        sample_ms = [float(value) / 1_000_000 for value in nanoseconds]
        sample_tokens_per_second = [token_count * 1_000_000_000 / float(value) for value in nanoseconds]
        parsed[phase] = {
            "tokens_per_sample": token_count,
            "duration_ms_samples": sample_ms,
            "tokens_per_second_samples": sample_tokens_per_second,
            "reported_tokens_per_second_samples": rates,
            "median_duration_ms": statistics.median(sample_ms),
            "median_tokens_per_second": statistics.median(sample_tokens_per_second),
            "llama_bench_metadata": {key: value for key, value in row.items() if key not in {"samples_ns", "samples_ts"}},
        }
    if set(parsed) != {"prefill", "decode"}:
        raise ValueError("llama-bench 缺少 prefill 或 decode 测试")
    return parsed


def run_mode(
    mode: str,
    binary: Path,
    model: Path,
    arguments: argparse.Namespace,
    nvidia_smi: str | None,
) -> dict[str, Any]:
    """启动一个 llama-bench 进程并独立采样其内存和显存。"""
    gpu_layers = 0 if mode == "cpu" else arguments.gpu_layers
    command = [
        str(binary), "-m", str(model), "-p", str(arguments.prompt_tokens),
        "-n", str(arguments.decode_tokens), "-r", str(arguments.repetitions),
        "-t", str(arguments.threads), "-b", str(arguments.prompt_tokens),
        "-ub", str(arguments.prompt_tokens), "-ngl", str(gpu_layers),
        "-o", "json",
    ]
    # 详细日志用于确认实际层分配和 CPU/GPU 缓冲区，不能只凭 -ngl 参数判断执行位置。
    command.append("-v")
    if mode == "gpu":
        command.extend(("-dev", arguments.gpu_device))
    if arguments.no_warmup:
        command.append("--no-warmup")

    # 启动前采样全局显存作为基线；它包含别的程序，不能称作进程显存。
    baseline_gpu = gpu_by_index(nvidia_smi, arguments.gpu_index)
    stop_monitor = threading.Event()
    sample_lock = threading.Lock()
    peak_rss_bytes = 0
    peak_global_vram_mib = baseline_gpu["memory_used_mib"] if baseline_gpu else None
    peak_process_vram_mib: float | None = None
    started = time.perf_counter()
    with subprocess.Popen(
        command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        observed = psutil.Process(process.pid)

        def monitor_rss() -> None:
            """以较短间隔记录 llama-bench 进程的峰值常驻内存。"""
            nonlocal peak_rss_bytes
            while not stop_monitor.is_set():
                try:
                    observed_rss = observed.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                with sample_lock:
                    peak_rss_bytes = max(peak_rss_bytes, observed_rss)
                stop_monitor.wait(arguments.rss_sample_seconds)

        def monitor_vram() -> None:
            """记录全局显存峰值，并在驱动支持时记录本进程显存。"""
            nonlocal peak_global_vram_mib, peak_process_vram_mib
            if nvidia_smi is None:
                return
            while not stop_monitor.is_set():
                snapshot = gpu_by_index(nvidia_smi, arguments.gpu_index)
                if snapshot is not None:
                    with sample_lock:
                        peak_global_vram_mib = max(
                            peak_global_vram_mib or 0, snapshot["memory_used_mib"]
                        )
                process_memory = gpu_process_used_mib(nvidia_smi, process.pid)
                if process_memory is not None:
                    with sample_lock:
                        peak_process_vram_mib = max(peak_process_vram_mib or 0, process_memory)
                stop_monitor.wait(arguments.gpu_sample_seconds)

        # 两种采样并行，避免 nvidia-smi 的轮询延迟遮蔽 RAM 峰值。
        monitors = [threading.Thread(target=monitor_rss, daemon=True)]
        if nvidia_smi is not None:
            monitors.append(threading.Thread(target=monitor_vram, daemon=True))
        for monitor in monitors:
            monitor.start()
        try:
            stdout, stderr = process.communicate()
        finally:
            stop_monitor.set()
            for monitor in monitors:
                monitor.join(timeout=7)
    wall_seconds = time.perf_counter() - started
    final_gpu = gpu_by_index(nvidia_smi, arguments.gpu_index)
    if process.returncode != 0:
        raise RuntimeError(
            f"{mode} llama-bench 退出码 {process.returncode}；stderr 末尾：\n"
            + "\n".join(stderr.splitlines()[-40:])
        )
    phases = benchmark_rows(
        stdout, arguments.repetitions, arguments.prompt_tokens, arguments.decode_tokens
    )
    backends = str(phases["prefill"]["llama_bench_metadata"].get("backends", ""))
    if mode == "gpu" and (not backends or backends.upper() == "CPU"):
        raise ValueError(f"GPU 测试并未报告 GPU 后端：{backends!r}")
    if mode == "cpu" and int(phases["prefill"]["llama_bench_metadata"].get("n_gpu_layers", -1)) != 0:
        raise ValueError("CPU 测试的 n_gpu_layers 并非 0")

    # 上游日志可辅助判断实际卸载层数；没有此日志时保留空值。
    offload_match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers?\s+to\s+GPU", stderr, re.I)
    offloaded_layers = int(offload_match.group(1)) if offload_match else None
    total_layers = int(offload_match.group(2)) if offload_match else None
    # 层被分配到 GPU 不代表全部量化矩阵也常驻显存，须单独记录实际缓冲区。
    buffer_matches = re.findall(
        r"load_tensors:\s+([A-Za-z0-9_]+) model buffer size\s*=\s*([0-9.]+) MiB",
        stderr,
    )
    model_buffers_mib = {name: float(size) for name, size in buffer_matches}
    if mode == "gpu" and offloaded_layers == 0:
        raise ValueError("GPU 后端存在，但实际卸载层数为 0")

    # WDDM 经常不报告 Vulkan 的进程级显存，此时只提供全局增量及其限制。
    global_delta = None
    if baseline_gpu is not None and peak_global_vram_mib is not None:
        global_delta = max(0.0, peak_global_vram_mib - baseline_gpu["memory_used_mib"])
    warnings: list[str] = []
    if mode == "gpu" and offloaded_layers is None:
        warnings.append("未从 stderr 提取到卸载层数；需复核 GPU 实际卸载")
    if peak_process_vram_mib is None and mode == "gpu":
        warnings.append("nvidia-smi 未提供 Vulkan 进程显存；全局增量可能包含其他进程")
    if mode == "gpu" and global_delta == 0 and peak_process_vram_mib is None:
        warnings.append("未观察到显存增量；请勿将此项作为可靠 GPU 占用数据")
    if baseline_gpu and final_gpu and abs(final_gpu["memory_used_mib"] - baseline_gpu["memory_used_mib"]) > 128:
        warnings.append("运行前后全局显存基线相差超过 128 MiB；可能有其他进程干扰")

    return {
        "mode": mode,
        "binary": str(binary),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": file_sha256(binary),
        "command": command,
        "wall_seconds_including_load": wall_seconds,
        "process_peak_rss_mib": peak_rss_bytes / MIB,
        "gpu": {
            "baseline": baseline_gpu,
            "final": final_gpu,
            "peak_global_used_mib": peak_global_vram_mib,
            "peak_global_delta_mib": global_delta,
            "peak_process_used_mib": peak_process_vram_mib,
            "offloaded_layers": offloaded_layers,
            "total_layers": total_layers,
            "model_buffers_mib": model_buffers_mib,
        },
        "phases": phases,
        "warnings": warnings,
        "stderr_tail": stderr.splitlines()[-100:],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """把本地原始记录写在 test 目录，避免误将模型路径或结果上传。"""
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """解析命令、核对模型来源并顺序完成 CPU/GPU 测量。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="官方 I2_S GGUF 的本地路径")
    parser.add_argument("--mode", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--cpu-binary", type=Path, help="CPU 版 llama-bench 可执行文件")
    parser.add_argument("--gpu-binary", type=Path, help="Vulkan/CUDA 版 llama-bench 可执行文件")
    parser.add_argument("--source-revision", default=DEFAULT_REVISION, help="官方 GGUF 仓库的 40 位提交")
    parser.add_argument("--output", type=Path, default=TEST_ROOT / "public_base_llama_bench.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-device", help="llama-bench --list-devices 列出的目标后端设备，例如 Vulkan1")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--rss-sample-seconds", type=float, default=0.05)
    parser.add_argument("--gpu-sample-seconds", type=float, default=0.25)
    arguments = parser.parse_args()

    # 防止公开报告引用其他项目检查点或不完整的下载文件。
    model = arguments.model.resolve(strict=True)
    if model.suffix.lower() != ".gguf" or model.stat().st_size < 100 * MIB:
        parser.error("--model 必须是完整的公开 GGUF 文件")
    with model.open("rb") as source:
        if source.read(4) != b"GGUF":
            parser.error("--model 文件没有 GGUF 魔数")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", arguments.source_revision):
        parser.error("--source-revision 必须是 40 位 Git 提交哈希")
    if arguments.source_revision.lower() != DEFAULT_REVISION:
        parser.error("本基准只接受已核对的微软原版模型提交")
    if model.stat().st_size != OFFICIAL_MODEL_BYTES:
        parser.error("GGUF 文件大小与微软原版模型不一致")
    # 必须检查文件内容，而非仅信任操作员填写的来源字段。
    model_sha256 = file_sha256(model)
    if model_sha256 != OFFICIAL_MODEL_SHA256:
        parser.error("GGUF SHA-256 与微软原版模型不一致")
    if any(value <= 0 for value in (
        arguments.threads, arguments.prompt_tokens, arguments.decode_tokens,
        arguments.repetitions, arguments.gpu_layers, arguments.rss_sample_seconds,
        arguments.gpu_sample_seconds,
    )):
        parser.error("线程数、样本尺寸、重复次数、卸载层数和采样间隔均须大于 0")
    if arguments.gpu_index < 0:
        parser.error("--gpu-index 不得为负数")
    output = arguments.output.resolve()
    if output.parent != TEST_ROOT.resolve() or output.suffix.lower() != ".json":
        parser.error("--output 必须是项目根目录 test/ 下的 JSON 文件")

    # 两种模式顺序执行，避免互相争用 CPU/GPU 和显存。
    modes = ("cpu", "gpu") if arguments.mode == "both" else (arguments.mode,)
    binaries: dict[str, Path] = {}
    for mode in modes:
        candidate = arguments.cpu_binary if mode == "cpu" else arguments.gpu_binary
        if candidate is None:
            parser.error(f"{mode} 模式缺少 --{mode}-binary")
        binary = candidate.resolve(strict=True)
        if not binary.is_file():
            parser.error(f"{mode} 二进制路径不是文件")
        binaries[mode] = binary

    nvidia_smi = shutil.which("nvidia-smi")
    if "gpu" in modes and not arguments.gpu_device:
        parser.error("GPU 模式需要 --gpu-device，避免自动选择错误的显卡")
    if "gpu" in modes and (nvidia_smi is None or gpu_by_index(nvidia_smi, arguments.gpu_index) is None):
        parser.error("GPU 模式需要 nvidia-smi 和有效的 --gpu-index")

    # 明确区分合成 prefill/decode 测试与 bit-jev 多问题判定任务。
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "benchmark_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "microsoft/bitnet-b1.58-2B-4T",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": arguments.source_revision.lower(),
        "source_revision_is_operator_assertion": True,
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": model_sha256,
            "format": "I2_S GGUF",
        },
        "workload": {
            "kind": "llama-bench synthetic backbone prefill and decode, not bit-jev decisions",
            "prompt_tokens": arguments.prompt_tokens,
            "decode_tokens": arguments.decode_tokens,
            "repetitions": arguments.repetitions,
            "threads": arguments.threads,
            "batch_tokens": arguments.prompt_tokens,
            "microbatch_tokens": arguments.prompt_tokens,
            "warmup_per_phase": not arguments.no_warmup,
            "model_load_excluded_from_throughput": True,
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu_name": cpu_name(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "system_ram_mib": psutil.virtual_memory().total / MIB,
            "gpu": gpu_by_index(nvidia_smi, arguments.gpu_index),
        },
        "source": {
            "project_git_commit": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
            "project_worktree_clean": git_value(PROJECT_ROOT, "status", "--porcelain") == "",
            "upstream_git_commit": git_value(PROJECT_ROOT / "learning" / "bitnet", "rev-parse", "HEAD"),
            "benchmark_compat_patch_sha256": file_sha256(TEST_ROOT / "patches" / "public-bitnet25-compat.patch"),
        },
        "runs": {},
        "measurement_notes": [
            "llama-bench -p 与 -n 分别产生 prefill 和 decode 测试，不是同一道题的端到端时延。",
            "吞吐样本排除模型加载；wall_seconds_including_load 包含加载和全部测试，仅供参考。",
            "CPU 内存是进程峰值 RSS；GPU 显存优先记录进程值，若驱动未提供则仅记录全局基线增量。",
        ],
    }
    write_report(output, report)
    for mode in modes:
        try:
            report["runs"][mode] = run_mode(mode, binaries[mode], model, arguments, nvidia_smi)
            write_report(output, report)
            rate = report["runs"][mode]["phases"]["prefill"]["median_tokens_per_second"]
            print(f"{mode}: prefill 中位吞吐 {rate:.2f} token/s", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            report["status"] = "failed"
            report["error"] = f"{mode}: {error}"
            write_report(output, report)
            print(report["error"], file=sys.stderr, flush=True)
            return 1
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(output, report)
    print(f"原始报告：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
