"""根据脱敏的公开单题案例数据生成中英文性能图。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


# 输入仅使用已审阅并纳入仓库的脱敏汇总数据。
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "benchmark-data" / "bit-jev-autodl-case-2026-09-27.json"
FIGURE_DIR = ROOT / "docs" / "figures"

# 图表在中英文版本中使用一致的颜色和字号。
CPU_COLOR = "#2563EB"
GPU_COLOR = "#7C3AED"
INK = "#172033"
MUTED = "#53627A"
GRID = "#DDE4ED"
PAPER = "#F5F7FB"


def read_and_validate() -> dict[str, Any]:
    """读取公开测量并核对样本形状、硬件和模型摘要。"""
    # 图表禁止从本机原始请求、预测输出或权重路径读取信息。
    record = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if record["publication_status"] != "public_sanitized_case_study":
        raise ValueError("公开单题案例状态不匹配")
    if record["workload"] != {
        "questions": 1,
        "input_tokens": 703,
        "options": 77,
        "input_contents_included": False,
    }:
        raise ValueError("工作负载形状不匹配")
    if record["checkpoint"]["cpu_gguf_sha256"] != "731700ba3112e35dc9bbecf54fe97b79d91b69246d765413ce683e06250773ff":
        raise ValueError("I2_S GGUF 摘要不匹配")
    if len(record["measurements"]) != 3:
        raise ValueError("CPU 8 线程、CPU 16 线程与 GPU 三条路径必须齐全")
    # 各路径应包含足够的正数样本，防止错误数据生成误导性图表。
    for measurement in record["measurements"]:
        samples = measurement["inference_ms"]
        if len(samples) < 3 or any(value <= 0 for value in samples):
            raise ValueError(f"{measurement['path']} 的计时样本无效")
    return record


def localized_text(language: str) -> dict[str, str]:
    """返回单一语言的标题、轴标签和说明文字。"""
    # 图中文字与文件名保持一致，便于 README 和模型卡独立引用。
    if language == "zh-CN":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL 单题实测案例",
            "title": "一道题：延迟与内存实测",
            "subtitle": "Xeon Gold 6459C + RTX 5090  ·  703 输入 tokens  ·  77 个候选项",
            "latency": "平均推理延迟（毫秒，对数刻度）",
            "memory": "峰值内存观测（MiB）",
            "cpu8": "CPU · 8 线程\nI2_S 原生",
            "cpu16": "CPU · 16 线程\nI2_S 原生",
            "gpu": "RTX 5090\nFP16 混合精度",
            "foot1": "计时不含模型加载；GPU 计时也排除预热。CPU/GPU 格式与数值精度不同。",
            "foot2": "CPU RSS 与 GPU 框架分配不是同一内存口径。此开发题与少量重复不代表通用性能。",
            "source": "公开脱敏数据：docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "毫秒",
        }
    if language == "en":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL SINGLE-QUESTION CASE",
            "title": "One question: latency and memory",
            "subtitle": "Xeon Gold 6459C + RTX 5090  ·  703 input tokens  ·  77 options",
            "latency": "Mean inference latency (ms, log scale)",
            "memory": "Peak memory observation (MiB)",
            "cpu8": "CPU · 8 threads\nNative I2_S",
            "cpu16": "CPU · 16 threads\nNative I2_S",
            "gpu": "RTX 5090\nFP16 mixed precision",
            "foot1": "Latency excludes model load; GPU timing also excludes warmup. CPU/GPU formats and precision differ.",
            "foot2": "CPU RSS and GPU framework allocation are different memory measures. This small development case is not general performance.",
            "source": "Sanitized public data: docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "milliseconds",
        }
    raise ValueError(f"不支持的图表语言：{language}")


def configure_fonts() -> None:
    """按当前系统查找简体中文字体，并保留跨平台回退。"""
    # 依次检查 Windows、Linux 与 macOS 常见字体位置，不把本机字体打包进仓库。
    candidates = (
        Path("C:/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    # 找到一款字体后注册为首选，使中文导出为可读字形而非方框。
    for font_path in candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    # Unicode 负号按正常数学符号显示，避免字体缺字时出现方框。
    plt.rcParams["axes.unicode_minus"] = False


def render(language: str, record: dict[str, Any]) -> None:
    """绘制指定语言的延迟与内存双面板图并导出 PNG、SVG。"""
    # 从逐次样本计算均值，图表数值始终与公开 JSON 同步。
    texts = localized_text(language)
    measurements = record["measurements"]
    latency_ms = [mean(row["inference_ms"]) for row in measurements]
    memory_mib = [
        measurements[0]["peak_process_rss_mib"],
        measurements[1]["peak_process_rss_mib"],
        measurements[2]["peak_gpu_allocation_mib"],
    ]
    labels = [texts["cpu8"], texts["cpu16"], texts["gpu"]]
    colors = [CPU_COLOR, CPU_COLOR, GPU_COLOR]

    # 统一画布并留出脚注区，图表可在论文 README 与社交媒体中复用。
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.875), dpi=180, facecolor=PAPER)
    figure.subplots_adjust(left=0.16, right=0.96, top=0.76, bottom=0.30, wspace=0.55)
    figure.text(0.06, 0.94, texts["eyebrow"], fontsize=11, fontweight="bold", color=CPU_COLOR)
    figure.text(0.06, 0.875, texts["title"], fontsize=25, fontweight="bold", color=INK)
    figure.text(0.06, 0.825, texts["subtitle"], fontsize=11, color=MUTED)

    # 分别使用独立坐标轴；延迟使用对数尺度，避免 GPU 短条不可读。
    axes[0].barh(range(3), latency_ms, color=colors, height=0.52)
    axes[0].set_xscale("log")
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(texts["unit"], color=MUTED, labelpad=9)
    axes[0].set_title(texts["latency"], loc="left", color=INK, fontsize=13, fontweight="bold", pad=15)
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    for position, value in enumerate(latency_ms):
        # 每根条末端直接标明算术平均值，避免仅凭相对长度估读。
        axes[0].text(value * 1.08, position, f"{value:,.1f} ms", va="center", fontsize=11, color=INK, fontweight="bold")

    # 内存图展示各路径的记录值，图注会说明 CPU 与 GPU 的计量定义不同。
    axes[1].barh(range(3), memory_mib, color=colors, height=0.52)
    axes[1].set_yticks(range(3), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MiB", color=MUTED, labelpad=9)
    axes[1].set_title(texts["memory"], loc="left", color=INK, fontsize=13, fontweight="bold", pad=15)
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].set_xlim(0, max(memory_mib) * 1.28)
    for position, value in enumerate(memory_mib):
        # 标签交代观测值，单位与坐标轴统一为 MiB。
        axes[1].text(value + max(memory_mib) * 0.025, position, f"{value:,.0f} MiB", va="center", fontsize=11, color=INK, fontweight="bold")

    # 统一简化图框，并保留对照色图例。
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=9)
    figure.text(0.06, 0.205, texts["foot1"], fontsize=9, color=INK)
    figure.text(0.06, 0.16, texts["foot2"], fontsize=9, color=MUTED)
    figure.text(0.06, 0.08, texts["source"], fontsize=8, color=MUTED)

    # 两种格式共用画布和数据，避免 PNG 与 SVG 出现内容差异。
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"bit-jev-autodl-case.{language}"
    for extension in ("png", "svg"):
        # PNG 用于 GitHub 预览，SVG 保留可缩放的文字和路径。
        destination = FIGURE_DIR / f"{stem}.{extension}"
        figure.savefig(destination, dpi=180, facecolor=PAPER)
        if extension == "svg":
            # 去掉 Matplotlib SVG 行尾空格，保证 Git diff 稳定。
            lines = destination.read_text(encoding="utf-8").splitlines()
            destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def main() -> None:
    """校验唯一公开数据源并生成中英两套图。"""
    # 同一份数据分别传给两种语言，保证数值完全一致。
    configure_fonts()
    record = read_and_validate()
    for language in ("zh-CN", "en"):
        render(language, record)
    print(f"已在 {FIGURE_DIR} 生成中英文 AutoDL 单题图")


if __name__ == "__main__":
    main()
