"""从已保存的基准测试报告生成可公开分享的性能图。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# 项目路径统一以脚本的位置为基准，避免受执行目录影响。
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "test"
FIGURE_DIR = ROOT / "docs" / "figures"

# 两张图统一使用适合 README 和社交媒体的配色。
BACKGROUND = "#F5F7FB"
INK = "#172033"
MUTED = "#53627A"
HAIRLINE = "#DDE4ED"
BLUE = "#2563EB"
PALE_BLUE = "#DBEAFE"
SLATE = "#94A3B8"
PALE_SLATE = "#E2E8F0"
WHITE = "#FFFFFF"


def read_report(name: str) -> dict:
    """以 UTF-8 读取一份现有测试报告。"""
    # 报告文件名来自项目内固定的基准测试产物。
    report_path = REPORT_DIR / name
    return json.loads(report_path.read_text(encoding="utf-8"))


def add_round_rect(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    radius: float = 0.018,
    edge_color: str | None = None,
) -> None:
    """在画布坐标中绘制带圆角的卡片或数据条。"""
    # 所有坐标都采用整张图片的比例，保证 SVG 与 PNG 布局一致。
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=color,
        edgecolor=edge_color if edge_color else "none",
        linewidth=1 if edge_color else 0,
    )
    fig.patches.append(patch)


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
    """在画布坐标中加入一段文字。"""
    fig.text(
        x,
        y,
        value,
        transform=fig.transFigure,
        fontsize=size,
        color=color,
        fontweight=weight,
        horizontalalignment=align,
        verticalalignment="center",
        family="DejaVu Sans",
    )


def new_figure() -> plt.Figure:
    """创建适合 README 和 X 配图的 16:9 画布。"""
    # 保留浅色背景，方便在 GitHub 的明暗主题中辨认图表。
    fig = plt.figure(figsize=(13.33, 7.5), dpi=180, facecolor=BACKGROUND)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    """将同一画布导出为可编辑 SVG 和高分辨率 PNG。"""
    # 采用可选择的 SVG 文本，方便后续复核具体测量值。
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        # 两种格式共用同一个数据源和排版，避免内容分叉。
        output_path = FIGURE_DIR / f"{stem}.{extension}"
        fig.savefig(
            output_path,
            dpi=180,
            facecolor=BACKGROUND,
        )
        if extension == "svg":
            # Matplotlib 会在路径数据行尾留下空格；清除它们以保持 Git 差异干净。
            svg_lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    plt.close(fig)


def metric_card(
    fig: plt.Figure,
    x: float,
    heading: str,
    takeaway: str,
    unit: str,
    i2s_value: float,
    f16_value: float,
    i2s_label: str,
    f16_label: str,
) -> None:
    """绘制同权重对比图中的一组指标。"""
    # 两根条在各自指标内按 F16 数值归一化；完整数值直接标在右侧。
    card_width = 0.275
    bar_width = 0.215
    add_round_rect(fig, x, 0.230, card_width, 0.495, WHITE, edge_color=HAIRLINE)
    add_text(fig, x + 0.025, 0.675, heading, 13, MUTED, "bold")
    add_text(fig, x + 0.025, 0.594, takeaway, 20, INK, "bold")
    add_text(fig, x + 0.025, 0.545, f"Lower {unit} is better", 10, MUTED)

    # I2_S 以蓝色表示；F16 以中性灰表示，避免暗示质量评分。
    for label, value, y, bar_color in (
        ("I2_S", i2s_value, 0.453, BLUE),
        ("F16", f16_value, 0.332, SLATE),
    ):
        add_text(fig, x + 0.025, y + 0.049, label, 12, INK, "bold")
        add_text(fig, x + card_width - 0.025, y + 0.049, (
            i2s_label if label == "I2_S" else f16_label
        ), 12, INK, "bold", "right")
        add_round_rect(fig, x + 0.025, y, bar_width, 0.025, PALE_SLATE, 0.012)
        add_round_rect(
            fig,
            x + 0.025,
            y,
            bar_width * value / f16_value,
            0.025,
            bar_color,
            0.012,
        )


def render_cpu_format() -> None:
    """制作同一权重的 I2_S 与 F16 原生 CPU 对照图。"""
    # 每个图上指标都直接取自同一份原始汇总报告。
    report = read_report("cpu_i2s_vs_f16_report.json")
    configuration = report["configuration"]
    i2s = report["i2s"]
    f16 = report["f16"]

    # 这些约束防止工作负载被更换后沿用旧的图标题和脚注。
    assert configuration["threads"] == 8
    assert configuration["batch"] == 128
    assert configuration["records_per_round"] == 12
    assert configuration["questions_per_round"] == 23
    assert configuration["repeats"] == 3
    assert len(report["prediction_flips"]) == 0

    # 文件体积用十进制 GB；进程内存保留报告中的二进制 MiB。
    i2s_file_gb = i2s["model_bytes"] / 1_000_000_000
    f16_file_gb = f16["model_bytes"] / 1_000_000_000
    i2s_rss_mib = i2s["peak_rss_mib"]
    f16_rss_mib = f16["peak_rss_mib"]
    i2s_time_s = mean(i2s["round_ms"]) / 1000
    f16_time_s = mean(f16["round_ms"]) / 1000

    # 重新计算比值并与汇总字段核对，避免展示旧的手填数字。
    assert abs(f16_file_gb / i2s_file_gb - report["f16_file_size_multiple_of_i2s"]) < 1e-9
    assert abs(f16_rss_mib / i2s_rss_mib - report["f16_peak_rss_multiple_of_i2s"]) < 1e-9
    assert abs(f16_time_s / i2s_time_s - report["i2s_inference_speedup_over_f16"]) < 1e-9

    fig = new_figure()
    add_text(fig, 0.055, 0.920, "BIT-JEV  /  NATIVE CPU BENCHMARK", 12, BLUE, "bold")
    add_text(fig, 0.055, 0.842, "Same weights. Less memory. Faster CPU inference.", 26, INK, "bold")
    add_text(
        fig,
        0.055,
        0.780,
        "I2_S GGUF vs F16 unpacked from that GGUF, using the same native runner and pointer head.",
        12,
        MUTED,
    )

    # 指标卡逐一展示原始量纲、绝对数值和可复核的改善倍数。
    metric_card(
        fig,
        0.055,
        "BACKBONE FILE",
        f"{f16_file_gb / i2s_file_gb:.2f}x smaller",
        "GB",
        i2s_file_gb,
        f16_file_gb,
        f"{i2s_file_gb:.2f} GB",
        f"{f16_file_gb:.2f} GB",
    )
    metric_card(
        fig,
        0.363,
        "PEAK PROCESS RSS",
        f"{f16_rss_mib / i2s_rss_mib:.2f}x lower",
        "MiB",
        i2s_rss_mib,
        f16_rss_mib,
        f"{i2s_rss_mib:,.0f} MiB",
        f"{f16_rss_mib:,.0f} MiB",
    )
    metric_card(
        fig,
        0.671,
        "NATIVE TIME / ROUND",
        f"{f16_time_s / i2s_time_s:.2f}x faster",
        "seconds",
        i2s_time_s,
        f16_time_s,
        f"{i2s_time_s:.2f} s",
        f"{f16_time_s:.2f} s",
    )

    # 发布图本身带齐测试条件，不依赖帖文才能正确解释。
    add_text(
        fig,
        0.055,
        0.169,
        "Ryzen 7 4800H  ·  8 CPU threads  ·  batch 128  ·  12 requests / 23 questions per round  ·  3 rounds",
        11,
        INK,
    )
    add_text(
        fig,
        0.055,
        0.117,
        "Latency excludes model load. 0 prediction flips across 69 outputs; max absolute logit difference 0.0689.",
        10,
        MUTED,
    )
    add_text(fig, 0.055, 0.064, "Source: test/cpu_i2s_vs_f16_report.json  ·  bit-jev", 9, MUTED)
    save_figure(fig, "cpu-format-comparison")


def single_question_card(
    fig: plt.Figure,
    x: float,
    heading: str,
    subheading: str,
    time_label: str,
    speed_label: str,
    memory_label: str,
    repeats_label: str,
    accent: str,
) -> None:
    """绘制固定单题案例的一个设备路径。"""
    # 每条路径展示延迟、输入吞吐和占用，不把两种精度混成同一测试条件。
    card_width = 0.275
    add_round_rect(fig, x, 0.287, card_width, 0.431, WHITE, edge_color=HAIRLINE)
    add_round_rect(fig, x + 0.025, 0.668, 0.045, 0.007, accent, 0.003)
    add_text(fig, x + 0.025, 0.630, heading, 15, INK, "bold")
    add_text(fig, x + 0.025, 0.587, subheading, 10, MUTED)
    add_text(fig, x + 0.025, 0.506, time_label, 28, accent, "bold")
    add_text(fig, x + 0.025, 0.462, "mean inference latency", 10, MUTED)
    add_text(fig, x + 0.025, 0.398, speed_label, 15, INK, "bold")
    add_text(fig, x + 0.025, 0.365, "input tokens / second", 10, MUTED)
    add_text(fig, x + 0.025, 0.323, f"{memory_label}  ·  {repeats_label}", 9, MUTED)


def render_single_question() -> None:
    """制作 77 选项单题的 CPU 和 GPU 部署路径案例图。"""
    # CPU 为原生 I2_S，GPU 为实验性混合精度；两者必须分别标明。
    cpu_8 = read_report("cpu_5090host_8threads_report.json")
    cpu_16 = read_report("cpu_5090host_16threads_report.json")
    gpu = read_report("gpu_5090_one_question.json")
    encoded_request = json.loads(
        (REPORT_DIR / "native_one_question_three.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    # 读取实际编码结果，保证“77 选项 / 703 token”不是手填的图注。
    row = encoded_request["rows"][0]
    input_tokens = len(row["ids"])
    options = len(row["options"])
    assert (input_tokens, options) == (703, 77)
    assert cpu_8["threads"] == 8 and cpu_16["threads"] == 16
    assert cpu_8["records"] == cpu_16["records"] == 3
    assert gpu["records"] == gpu["questions"] == 1 and gpu["repeats"] == 5
    assert gpu["device"] == "NVIDIA GeForce RTX 5090"

    # CPU 原生延迟排除加载；GPU 采用报告中预热后的逐次前向延迟。
    cpu_8_ms = mean(cpu_8["native_latency_ms"])
    cpu_16_ms = mean(cpu_16["native_latency_ms"])
    gpu_ms = mean(gpu["inference_ms_by_repeat"])
    assert abs(gpu_ms - gpu["inference_ms_mean_per_repeat"]) < 1e-9
    speedup = cpu_16_ms / gpu_ms

    fig = new_figure()
    add_text(fig, 0.055, 0.920, "BIT-JEV  /  SINGLE-QUESTION CASE STUDY", 12, BLUE, "bold")
    add_text(fig, 0.055, 0.843, f"{input_tokens} input tokens. {options} options. One decision.", 27, INK, "bold")
    add_text(
        fig,
        0.055,
        0.779,
        "One fixed development question on a Xeon Gold 6459C host with an RTX 5090.",
        12,
        MUTED,
    )

    # 三个设备卡由对应的原始 JSON 报告直接填充。
    single_question_card(
        fig,
        0.055,
        "CPU · 8 threads",
        "I2_S · native runner",
        f"{cpu_8_ms / 1000:.2f} s",
        f"{input_tokens * 1000 / cpu_8_ms:,.0f}",
        f"{cpu_8['peak_rss_mib']:,.0f} MiB RSS",
        "n=3",
        BLUE,
    )
    single_question_card(
        fig,
        0.363,
        "CPU · 16 threads",
        "I2_S · native runner",
        f"{cpu_16_ms / 1000:.2f} s",
        f"{input_tokens * 1000 / cpu_16_ms:,.0f}",
        f"{cpu_16['peak_rss_mib']:,.0f} MiB RSS",
        "n=3",
        BLUE,
    )
    single_question_card(
        fig,
        0.671,
        "GPU · RTX 5090",
        "FP16 weights · mixed precision",
        f"{gpu_ms:.1f} ms",
        f"{input_tokens * 1000 / gpu_ms:,.0f}",
        f"{gpu['peak_memory_mib']:,.0f} MiB VRAM",
        "n=5",
        "#7C3AED",
    )

    # 脚注明确一个案例和不同数值路径，避免跨硬件误读。
    add_text(
        fig,
        0.055,
        0.227,
        f"GPU was {speedup:.1f}x faster than 16-thread CPU on this one question.",
        15,
        INK,
        "bold",
    )
    add_text(
        fig,
        0.055,
        0.158,
        "Latency excludes model load and GPU warmup. CPU and GPU use different weight and arithmetic formats.",
        10,
        MUTED,
    )
    add_text(
        fig,
        0.055,
        0.110,
        "Rates count input tokens, not generated output tokens. This single case is not a general speed claim.",
        10,
        MUTED,
    )
    add_text(
        fig,
        0.055,
        0.058,
        "Sources: test/cpu_5090host_{8,16}threads_report.json  ·  test/gpu_5090_one_question.json",
        9,
        MUTED,
    )
    save_figure(fig, "single-question-hardware")


def main() -> None:
    """生成两张由报告驱动的发布图。"""
    # 每张图单独读取报告，便于以后更新实验数据后重新生成。
    render_cpu_format()
    render_single_question()
    print(f"Created benchmark figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
