"""从独立实测报告生成微软公开 BitNet 基础模型的对比图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# 图表颜色在两张图中保持一致，CPU 为蓝色，GPU 为紫色。
BACKGROUND = "#F5F7FB"
CARD = "#FFFFFF"
INK = "#142033"
MUTED = "#526179"
HAIRLINE = "#DCE3EC"
TRACK = "#E9EEF5"
CPU_BLUE = "#2563EB"
GPU_PURPLE = "#7C3AED"

# 公开报告必须明确指向微软基础模型，防止误写为 bit-jev 蒸馏模型。
MODEL_FAMILY = "microsoft/bitnet-b1.58-2B-4T"
OFFICIAL_REPOSITORIES = {
    "CPU": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "GPU": "microsoft/bitnet-b1.58-2B-4T-bf16",
}


@dataclass(frozen=True)
class Configuration:
    """保存一条通过校验的部署路径及其原始观测值。"""

    device: str
    repository: str
    revision: str
    format: str
    hardware: str
    runtime: str
    operation: str
    memory_method: str
    warm_latency_ms: tuple[float, ...]
    loaded_memory_mib: float
    memory_capacity_mib: float | None
    model_file_bytes: int | None


@dataclass(frozen=True)
class Benchmark:
    """保存两条路径共用的工作负载和可追溯来源。"""

    benchmark_utc: str
    throughput_token_kind: str
    performance_definition: str
    input_tokens: int
    output_tokens: int
    warmup_requests: int
    timed_requests: int
    request_description: str
    cpu: Configuration
    gpu: Configuration
    report_name: str
    report_sha256: str

    @property
    def measured_tokens(self) -> int:
        """按工作负载类型选择实际计速的输入或输出 token 数。"""
        return self.input_tokens if self.throughput_token_kind == "input" else self.output_tokens

    @property
    def token_kind(self) -> str:
        """为图表返回与计速分母一致的 token 说明。"""
        return self.throughput_token_kind


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    """拒绝缺少对象结构的字段。"""
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON 对象")
    return value


def require_text(value: Any, field: str) -> str:
    """读取非空文本，并拒绝无法准确显示的换行。"""
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} 必须是非空单行字符串")
    return value.strip()


def require_count(value: Any, field: str, minimum: int = 0) -> int:
    """读取非布尔整数计数并检查下界。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} 必须是大于等于 {minimum} 的整数")
    return value


def require_positive_number(value: Any, field: str) -> float:
    """读取严格为正的有限测量值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是正数")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} 必须是有限正数")
    return number


def read_configuration(raw: Any, device: str, timed_requests: int) -> Configuration:
    """校验单条官方基础模型的部署、延迟与加载后内存。"""
    # 设备字段与官方仓库一同校验，避免把本项目的私有检查点混入图表。
    item = require_mapping(raw, f"configurations[{device}]")
    observed_device = require_text(item.get("device"), "device")
    if observed_device != device:
        raise ValueError(f"配置顺序必须为 CPU、GPU；当前收到 {observed_device}")
    repository = require_text(item.get("repository"), f"{device}.repository")
    if repository != OFFICIAL_REPOSITORIES[device]:
        raise ValueError(f"{device}.repository 必须是微软公开基础模型官方仓库")
    revision = require_text(item.get("revision"), f"{device}.revision")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(f"{device}.revision 必须是固定的 40 位提交哈希")

    # 原始逐次延迟由报告提供；图中中位数和吞吐在渲染时计算。
    raw_latencies = item.get("warm_latency_ms")
    if not isinstance(raw_latencies, list) or len(raw_latencies) != timed_requests:
        raise ValueError(f"{device}.warm_latency_ms 长度必须等于 timed_requests")
    latencies = tuple(
        require_positive_number(value, f"{device}.warm_latency_ms[{index}]")
        for index, value in enumerate(raw_latencies)
    )

    # CPU 使用进程常驻内存，GPU 使用设备显存；两者不伪装成同一指标。
    memory_field = "loaded_process_rss_mib" if device == "CPU" else "loaded_gpu_vram_mib"
    loaded_memory = require_positive_number(item.get(memory_field), f"{device}.{memory_field}")
    memory_capacity = item.get("memory_capacity_mib")
    if memory_capacity is not None:
        memory_capacity = require_positive_number(memory_capacity, f"{device}.memory_capacity_mib")
        if loaded_memory > memory_capacity:
            raise ValueError(f"{device}.memory_capacity_mib 不得小于加载后占用")
    model_file_bytes = item.get("model_file_bytes")
    if model_file_bytes is not None:
        model_file_bytes = require_count(model_file_bytes, f"{device}.model_file_bytes", 1)

    return Configuration(
        device=device,
        repository=repository,
        revision=revision.lower(),
        format=require_text(item.get("format"), f"{device}.format"),
        hardware=require_text(item.get("hardware"), f"{device}.hardware"),
        runtime=require_text(item.get("runtime"), f"{device}.runtime"),
        operation=require_text(item.get("operation"), f"{device}.operation"),
        memory_method=require_text(item.get("memory_method"), f"{device}.memory_method"),
        warm_latency_ms=latencies,
        loaded_memory_mib=loaded_memory,
        memory_capacity_mib=memory_capacity,
        model_file_bytes=model_file_bytes,
    )


def read_benchmark(report_path: Path) -> Benchmark:
    """读取完整报告并在画图之前完成全部口径校验。"""
    # 散列取自输入原始字节，便于读者核对公开报告和图表是否对应。
    report_bytes = report_path.read_bytes()
    raw = require_mapping(json.loads(report_bytes.decode("utf-8")), "报告")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version 必须为 1")
    if raw.get("model_family") != MODEL_FAMILY:
        raise ValueError("model_family 必须明确指定微软公开 BitNet 基础模型")
    benchmark_utc = require_text(raw.get("benchmark_utc"), "benchmark_utc")
    try:
        parsed_time = datetime.fromisoformat(benchmark_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("benchmark_utc 必须为 ISO 8601 时间") from error
    if parsed_time.utcoffset() is None or parsed_time.utcoffset().total_seconds() != 0:
        raise ValueError("benchmark_utc 必须使用 UTC 时区")

    # 同一份工作负载是跨设备速度并列展示的必要条件。
    workload = require_mapping(raw.get("workload"), "workload")
    throughput_token_kind = require_text(workload.get("throughput_token_kind"), "workload.throughput_token_kind")
    if throughput_token_kind not in {"input", "output"}:
        raise ValueError("workload.throughput_token_kind 只能是 input 或 output")
    performance_definition = require_text(workload.get("performance_definition"), "workload.performance_definition")
    input_tokens = require_count(workload.get("input_tokens"), "workload.input_tokens", 1)
    output_tokens = require_count(workload.get("output_tokens"), "workload.output_tokens")
    if throughput_token_kind == "output" and output_tokens < 1:
        raise ValueError("输出 token 吞吐测试必须有实际输出 token")
    warmup_requests = require_count(workload.get("warmup_requests"), "workload.warmup_requests", 1)
    timed_requests = require_count(workload.get("timed_requests"), "workload.timed_requests", 3)
    if workload.get("load_excluded") is not True:
        raise ValueError("workload.load_excluded 必须为 true")
    request_description = require_text(workload.get("request_description"), "workload.request_description")

    # 两条配置须来自同一个报告；不得以预填的均值、比值或虚构图例代替样本。
    configurations = raw.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != 2:
        raise ValueError("configurations 必须恰好包含 CPU 和 GPU 两条独立实测")
    cpu = read_configuration(configurations[0], "CPU", timed_requests)
    gpu = read_configuration(configurations[1], "GPU", timed_requests)
    if cpu.format.upper() != "I2_S GGUF" or gpu.format.upper() not in {
        "BF16",
        "FP16 (CAST FROM BF16 CHECKPOINT)",
    }:
        raise ValueError("公开比较必须明确为 CPU I2_S GGUF，以及 GPU BF16 或 BF16 检查点转 FP16")
    return Benchmark(
        benchmark_utc=benchmark_utc,
        throughput_token_kind=throughput_token_kind,
        performance_definition=performance_definition,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        warmup_requests=warmup_requests,
        timed_requests=timed_requests,
        request_description=request_description,
        cpu=cpu,
        gpu=gpu,
        report_name=report_path.name,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )


def add_card(fig: plt.Figure, x: float, y: float, width: float, height: float) -> None:
    """绘制带细边框的图表卡片。"""
    # 卡片坐标使用整张画布比例，方便两种输出格式保持一致。
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.022",
        transform=fig.transFigure,
        facecolor=CARD,
        edgecolor=HAIRLINE,
        linewidth=1,
        zorder=0,
    )
    fig.patches.append(card)


def add_text(
    fig: plt.Figure,
    x: float,
    y: float,
    value: str,
    size: float,
    color: str = INK,
    weight: str = "normal",
    align: str = "left",
) -> None:
    """在整张画布上放置文本，避免各卡片字体基线漂移。"""
    fig.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=align,
        va="center",
        family="DejaVu Sans",
    )


def new_figure() -> plt.Figure:
    """建立用于 README 和 X 分享的统一宽屏画布。"""
    # SVG 保存为文字元素，保证公开图上的数值可以选择和检索。
    plt.rcParams["svg.fonttype"] = "none"
    return plt.figure(figsize=(13.33, 7.5), dpi=180, facecolor=BACKGROUND)


def add_heading(fig: plt.Figure, title: str, subtitle: str) -> None:
    """在每张图上保留醒目的来源与非 bit-jev 声明。"""
    add_text(fig, 0.055, 0.925, "PUBLIC MICROSOFT BITNET BASE  /  INDEPENDENT MEASUREMENT", 11, CPU_BLUE, "bold")
    add_text(fig, 0.055, 0.850, title, 25, INK, "bold")
    add_text(fig, 0.055, 0.795, subtitle, 11, MUTED)
    add_text(fig, 0.945, 0.925, "NOT A BIT-JEV CHECKPOINT", 10, GPU_PURPLE, "bold", "right")


def add_format_notice(fig: plt.Figure, benchmark: Benchmark) -> None:
    """突出标明两种精度路径，尤其是从 BF16 检查点转为 FP16 的情况。"""
    add_text(
        fig,
        0.055,
        0.752,
        f"CPU: {benchmark.cpu.format}  |  GPU: {benchmark.gpu.format}",
        10,
        GPU_PURPLE,
        "bold",
    )


def add_footer(fig: plt.Figure, benchmark: Benchmark, extra_lines: tuple[str, ...]) -> None:
    """为独立传播的图片补齐硬件、样本和原始文件线索。"""
    add_text(fig, 0.055, 0.205, f"CPU: {benchmark.cpu.hardware}  |  {benchmark.cpu.runtime}  |  {benchmark.cpu.operation}", 9, INK)
    add_text(fig, 0.055, 0.170, f"GPU: {benchmark.gpu.hardware}  |  {benchmark.gpu.runtime}  |  {benchmark.gpu.operation}", 9, INK)
    for index, line in enumerate(extra_lines[:2]):
        add_text(fig, 0.055, 0.125 - index * 0.033, line, 8.5, MUTED)
    add_text(
        fig,
        0.055,
        0.048,
        f"Raw: {benchmark.report_name}  |  SHA-256 {benchmark.report_sha256[:16]}  |  {benchmark.benchmark_utc}",
        8,
        MUTED,
    )


def add_two_bar_panel(
    fig: plt.Figure,
    x: float,
    title: str,
    unit: str,
    values: tuple[float, float],
    labels: tuple[str, str],
    lower_is_better: bool,
) -> None:
    """以独立的尺度画一组同量纲的 CPU/GPU 横条。"""
    add_card(fig, x, 0.270, 0.420, 0.455)
    add_text(fig, x + 0.030, 0.677, title, 14, INK, "bold")
    direction = "lower is better" if lower_is_better else "higher is better"
    add_text(fig, x + 0.030, 0.628, f"{unit}  ·  {direction}", 9, MUTED)
    axis = fig.add_axes((x + 0.132, 0.365, 0.230, 0.205), facecolor=CARD, zorder=2)
    for index, (value, color) in enumerate(((values[0], CPU_BLUE), (values[1], GPU_PURPLE))):
        y_position = 1 - index
        axis.barh(y_position, value, color=color, height=0.37)
        axis.text(value + max(values) * 0.025, y_position, f"{value:,.1f}", va="center", fontsize=10, color=INK)
    axis.set_xlim(0, max(values) * 1.34)
    axis.set_ylim(-0.6, 1.6)
    axis.set_yticks([1, 0], labels)
    axis.tick_params(axis="y", length=0, labelsize=10, colors=INK, pad=8)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_speed(benchmark: Benchmark) -> plt.Figure:
    """绘制热机单请求延迟和由相同延迟推算的 token 吞吐。"""
    # 中位数来自逐次样本；吞吐的分子固定为本报告的相同 token 工作量。
    cpu_latency = median(benchmark.cpu.warm_latency_ms)
    gpu_latency = median(benchmark.gpu.warm_latency_ms)
    cpu_rate = benchmark.measured_tokens * 1000 / cpu_latency
    gpu_rate = benchmark.measured_tokens * 1000 / gpu_latency
    fig = new_figure()
    add_heading(
        fig,
        "Warm request speed: CPU vs GPU",
        f"Same public base family  ·  {benchmark.request_description}  ·  model load excluded",
    )
    add_format_notice(fig, benchmark)
    gpu_label = "GPU FP16" if benchmark.gpu.format.startswith("FP16") else "GPU BF16"
    labels = ("CPU I2_S", gpu_label)
    add_two_bar_panel(fig, 0.055, "MEDIAN REQUEST LATENCY", "ms / request", (cpu_latency, gpu_latency), labels, True)
    add_two_bar_panel(
        fig,
        0.525,
        f"{benchmark.token_kind.upper()} TOKEN THROUGHPUT",
        f"{benchmark.token_kind} tokens / s",
        (cpu_rate, gpu_rate),
        labels,
        False,
    )
    add_footer(
        fig,
        benchmark,
        (
            f"Definition: {benchmark.performance_definition}  |  {benchmark.input_tokens} input / "
            f"{benchmark.output_tokens} output tokens",
            f"{benchmark.warmup_requests} warmups + {benchmark.timed_requests} timed requests  |  "
            "throughput = tokens / median request time",
        ),
    )
    return fig


def add_memory_panel(fig: plt.Figure, x: float, configuration: Configuration, metric: str) -> None:
    """在独立卡片中展示 CPU 进程内存或 GPU 设备显存。"""
    color = CPU_BLUE if configuration.device == "CPU" else GPU_PURPLE
    loaded_gib = configuration.loaded_memory_mib / 1024
    add_card(fig, x, 0.270, 0.420, 0.455)
    short_format = "FP16 cast" if configuration.format.startswith("FP16") else configuration.format
    add_text(fig, x + 0.030, 0.667, f"{configuration.device}  /  {short_format}", 14, INK, "bold")
    add_text(fig, x + 0.030, 0.605, metric, 10, MUTED)
    add_text(fig, x + 0.030, 0.515, f"{loaded_gib:,.2f} GiB", 29, color, "bold")
    add_text(fig, x + 0.030, 0.438, configuration.memory_method, 9, MUTED)
    # 有机器容量时，横条仅表示各自内存域内部的占用比例；缺失容量就不画数据条。
    if configuration.memory_capacity_mib is not None:
        bar_width = 0.345
        fraction = configuration.loaded_memory_mib / configuration.memory_capacity_mib
        track = FancyBboxPatch(
            (x + 0.030, 0.360),
            bar_width,
            0.029,
            boxstyle="round,pad=0,rounding_size=0.014",
            transform=fig.transFigure,
            facecolor=TRACK,
            edgecolor="none",
        )
        fill = FancyBboxPatch(
            (x + 0.030, 0.360),
            bar_width * fraction,
            0.029,
            boxstyle="round,pad=0,rounding_size=0.014",
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
        )
        fig.patches.extend((track, fill))
        capacity_gib = configuration.memory_capacity_mib / 1024
        add_text(fig, x + 0.030, 0.336, f"{fraction:.0%} of {capacity_gib:,.1f} GiB physical capacity", 8, MUTED)
    if configuration.model_file_bytes is not None:
        file_gb = configuration.model_file_bytes / 1_000_000_000
        add_text(fig, x + 0.030, 0.310, f"Artifact on disk: {file_gb:.2f} GB (decimal)", 9, MUTED)


def render_memory(benchmark: Benchmark) -> plt.Figure:
    """绘制两种部署格式加载完成后的独立内存指标。"""
    fig = new_figure()
    add_heading(
        fig,
        "Loaded model footprint: two deployment paths",
        "Same public base family  ·  measured after model load, before timed requests",
    )
    add_format_notice(fig, benchmark)
    add_memory_panel(fig, 0.055, benchmark.cpu, "PROCESS RESIDENT MEMORY (RSS)")
    add_memory_panel(fig, 0.525, benchmark.gpu, "GPU DEVICE MEMORY (VRAM)")
    add_footer(
        fig,
        benchmark,
        (
            "GiB = 1024 MiB. Process RSS and GPU VRAM are different memory domains.",
            "Capacity bars use each device's own physical limit when provided; no cross-device ratio is implied.",
        ),
    )
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """把同一张实测图保存为 SVG 和 PNG，并关闭画布。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("svg", "png"):
            output_path = output_dir / f"{stem}.{extension}"
            fig.savefig(output_path, dpi=180, facecolor=BACKGROUND)
            if extension == "svg":
                # 统一 SVG 行尾，避免无意义的版本控制差异。
                lines = output_path.read_text(encoding="utf-8").splitlines()
                output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    finally:
        plt.close(fig)


def main() -> None:
    """解析真实原始报告，并生成两张有出处的公开图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="官方基础模型独立实测 JSON 报告")
    parser.add_argument("--output-dir", type=Path, required=True, help="SVG 和 PNG 输出目录")
    args = parser.parse_args()
    # 输入不存在或报告不合规时直接退出，不能留下任何看似实测的图。
    try:
        benchmark = read_benchmark(args.report)
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(f"无法读取合规的实测报告：{error}")
    save_figure(render_speed(benchmark), args.output_dir, "public-base-speed")
    save_figure(render_memory(benchmark), args.output_dir, "public-base-loaded-memory")
    print(f"图表已生成：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
