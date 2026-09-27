"""把公开 BitNet 基础模型的实测转换成图表与可追溯数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 所有输入脚本与本地报告都位于项目根目录的 test 下。
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "test" / "public_base_llama_bench_final.json"
FIGURE_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "docs" / "benchmark-data"

# 两张图使用相同的部署路径颜色，便于直接对照。
CPU_COLOR = "#1672E8"
GPU_COLOR = "#8058DB"
INK = "#18243A"
MUTED = "#52627A"
LINE = "#DDE5EF"
PAPER = "#F5F8FC"


def sha256(path: Path) -> str:
    """计算原始报告摘要，供公开数据追溯本地记录。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """只接受指定微软公开模型、完整运行和相同测试负载。"""
    model = raw["model"]
    work = raw["workload"]
    runs = raw["runs"]
    if raw["status"] != "complete":
        raise ValueError("原始测量尚未完成")
    if raw["source_repository"] != "microsoft/bitnet-b1.58-2B-4T-gguf":
        raise ValueError("仅接受微软公开 BitNet GGUF")
    if raw["source_revision"] != "9f43072f69492cbd5bc5d5ebb085fec519686a93":
        raise ValueError("公开模型修订不匹配")
    if model["sha256"] != "13939ce5030319a35db346e5dba7a3a3bd599dfc18b113a2a97446ff964714c5":
        raise ValueError("模型 SHA-256 不匹配")
    if model["bytes"] != 1844472032 or model["format"] != "I2_S GGUF":
        raise ValueError("模型文件大小或格式不匹配")
    if (work["prompt_tokens"], work["decode_tokens"], work["repetitions"]) != (128, 32, 5):
        raise ValueError("工作负载与图表标注不匹配")
    if runs["cpu"]["mode"] != "cpu" or runs["gpu"]["mode"] != "gpu":
        raise ValueError("CPU/GPU 路径缺失")
    if runs["gpu"]["gpu"]["offloaded_layers"] != 31:
        raise ValueError("GPU 层卸载记录缺失")
    for device in ("cpu", "gpu"):
        for phase in ("prefill", "decode"):
            samples = runs[device]["phases"][phase]["duration_ms_samples"]
            if len(samples) != 5 or any(value <= 0 for value in samples):
                raise ValueError(f"{device}/{phase} 计时样本无效")
    return runs["cpu"], runs["gpu"]


def canvas(title: str, subtitle: str) -> tuple[Any, Any]:
    """创建带统一标题、留白和坐标样式的双面板画布。"""
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), facecolor=PAPER)
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.69, wspace=0.39)
    figure.text(0.075, 0.91, title, fontsize=22, fontweight="bold", color=INK)
    figure.text(0.075, 0.84, subtitle, fontsize=10.5, color=MUTED)
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(LINE)
        axis.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=10)
        axis.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
    return figure, axes


def bar_pair(axis: Any, title: str, values: tuple[float, float], unit: str) -> None:
    """在独立刻度上展示 CPU 与 GPU 数值，不隐藏较慢路径。"""
    maximum = max(values) * 1.25
    axis.set_title(title, fontsize=14, color=INK, fontweight="bold", loc="left", pad=18)
    axis.barh([1, 0], values, color=[CPU_COLOR, GPU_COLOR], height=0.36, zorder=3)
    axis.set_yticks([1, 0], ["CPU", "Vulkan hybrid"])
    axis.set_xlim(0, maximum)
    axis.set_xlabel(unit, color=MUTED, fontsize=10, labelpad=10)
    for position, value in zip((1, 0), values):
        axis.text(value + maximum * 0.018, position, f"{value:.2f}",
                  va="center", fontsize=12, fontweight="bold", color=INK)


def save(figure: Any, stem: str) -> None:
    """导出 GitHub 可显示的 SVG 与社交媒体可用的 PNG。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = FIGURE_DIR / f"{stem}.{extension}"
        figure.savefig(destination, dpi=180, facecolor=PAPER)
        if extension == "svg":
            # Matplotlib 在 SVG 路径数据行末留空格；清理后才能通过 Git 空白检查。
            lines = destination.read_text(encoding="utf-8").splitlines()
            destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def speed_figure(cpu: dict[str, Any], gpu: dict[str, Any]) -> None:
    """分开展示输入预填充与输出解码的吞吐和中位耗时。"""
    figure, axes = canvas("Public BitNet base: CPU vs GPU path",
                          "Microsoft original I2_S GGUF  |  Ryzen 7 4800H (8 threads)  |  RTX 2060 Vulkan")
    for axis, phase, title in zip(axes, ("prefill", "decode"),
                                  ("Prefill  |  128 input tokens", "Decode  |  32 output tokens")):
        values = (cpu["phases"][phase]["median_tokens_per_second"],
                  gpu["phases"][phase]["median_tokens_per_second"])
        bar_pair(axis, title, values, "Median throughput (tokens/s)")
    figure.text(0.075, 0.08,
                "5 repetitions per phase; model load excluded from throughput. GPU path is hybrid, not pure GPU.\n"
                "Synthetic llama-bench backbone test; not bit-jev decision latency or accuracy.",
                fontsize=9.3, color=MUTED, linespacing=1.7)
    save(figure, "public-base-speed")


def memory_figure(cpu: dict[str, Any], gpu: dict[str, Any]) -> None:
    """区分进程 RSS 和全局显存增量，不把不同内存口径相加。"""
    figure, axes = canvas("Public BitNet base: memory observed",
                          "Same I2_S GGUF on both paths  |  File on disk: 1,844,472,032 bytes (1.72 GiB)")
    bar_pair(axes[0], "Peak process RAM (RSS)",
             (cpu["process_peak_rss_mib"] / 1024, gpu["process_peak_rss_mib"] / 1024),
             "GiB, load + benchmark")
    axes[1].set_title("GPU global VRAM rise", fontsize=14, color=INK,
                      fontweight="bold", loc="left", pad=18)
    delta = gpu["gpu"]["peak_global_delta_mib"] / 1024
    axes[1].barh([0], [delta], color=GPU_COLOR, height=0.36, zorder=3)
    axes[1].set_yticks([0], ["Vulkan hybrid"])
    axes[1].set_ylim(-0.8, 0.8)
    axes[1].set_xlim(0, max(1.0, delta * 1.35))
    axes[1].set_xlabel("GiB over pre-run nvidia-smi baseline", color=MUTED, fontsize=10, labelpad=10)
    axes[1].text(delta + 0.025, 0, f"{delta:.2f}", va="center",
                 fontsize=12, fontweight="bold", color=INK)
    figure.text(0.075, 0.08,
                "RSS is process-wide peak sampled through model loading and tests. VRAM is global delta, not per-process allocation.\n"
                "Vulkan offloads 31/31 layers, but I2_S work stays partly CPU-mapped. RAM and VRAM are separate measures.",
                fontsize=9.1, color=MUTED, linespacing=1.7)
    save(figure, "public-base-memory")


def public_report(raw: dict[str, Any], report_hash: str) -> dict[str, Any]:
    """移除本机路径和冗长 stderr，保留逐次样本与计算来源。"""
    runs: dict[str, Any] = {}
    for device in ("cpu", "gpu"):
        run = raw["runs"][device]
        runs[device] = {
            "mode": run["mode"],
            "binary_sha256": run["binary_sha256"],
            "process_peak_rss_mib": run["process_peak_rss_mib"],
            "gpu": {
                "peak_global_delta_mib": run["gpu"]["peak_global_delta_mib"],
                "peak_process_used_mib": run["gpu"]["peak_process_used_mib"],
                "offloaded_layers": run["gpu"]["offloaded_layers"],
                "total_layers": run["gpu"]["total_layers"],
                "model_buffers_mib": run["gpu"].get("model_buffers_mib", {}),
            },
            "phases": run["phases"],
            "warnings": run["warnings"],
        }
    return {
        "schema_version": 1,
        "raw_report_sha256": report_hash,
        "benchmark_utc": raw["benchmark_utc"],
        "completed_utc": raw["completed_utc"],
        "model_family": raw["model_family"],
        "source_repository": raw["source_repository"],
        "source_revision": raw["source_revision"],
        "source_revision_is_operator_assertion": raw["source_revision_is_operator_assertion"],
        "model": {key: raw["model"][key] for key in ("bytes", "sha256", "format")},
        "workload": raw["workload"],
        "hardware": raw["hardware"],
        "source": raw["source"],
        "runs": runs,
        "measurement_notes": raw["measurement_notes"],
    }


def main() -> None:
    """校验报告，然后生成两张图与无本机路径的公开 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    cpu, gpu = validate(raw)
    speed_figure(cpu, gpu)
    memory_figure(cpu, gpu)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / "public-bitnet-base-2026-09-27.json"
    destination.write_text(json.dumps(public_report(raw, sha256(args.input)), ensure_ascii=False,
                                      indent=2) + "\n", encoding="utf-8")
    print(f"公开报告：{destination}")
    print(f"图表：{FIGURE_DIR / 'public-base-speed.svg'}，{FIGURE_DIR / 'public-base-memory.svg'}")


if __name__ == "__main__":
    main()
