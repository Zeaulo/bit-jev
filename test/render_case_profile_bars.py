"""把公开 AutoDL 单题案例画成适合 README 阅读的横向对比条。"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from render_checkpoint_public_figures import configure_fonts, read_and_validate


# 图片只从仓库中的脱敏公开记录读取，不能引入本机原始请求或预测结果。
ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
PAPER = "#FCFCFA"
INK = "#202328"
MUTED = "#636B75"
TRACK = "#C9CED3"
CPU_BLUE = "#1686C7"
CPU_TEAL = "#29A58A"
GPU_CORAL = "#E3695B"


def copy_for(language: str, kind: str) -> dict[str, str]:
    """为每种指标提供准确的中英文标题、路径标签与测量边界。"""
    # 延迟条越短越好；内存条同时展示不同测量口径，必须在图脚说明。
    if language == "zh-CN" and kind == "speed":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL 单题案例",
            "title": "推理耗时：越短越快",
            "subtitle": "同一开发题 · 703 输入 tokens · 77 个候选项 · 不含模型加载",
            "cpu8": "CPU · 8 线程 · I2_S",
            "cpu16": "CPU · 16 线程 · I2_S",
            "gpu": "RTX 5090 · FP16 混合精度",
            "foot1": "CPU 每条路径重复 3 次；GPU 重复 5 次，另排除预热。条长对应平均毫秒数。",
            "foot2": "CPU 与 GPU 的模型格式、精度不同；这是两套部署路径的单题案例，不代表纯硬件倍数。",
            "source": "数据：docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "ms",
        }
    if language == "zh-CN" and kind == "memory":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL 单题案例",
            "title": "峰值占用：越短越省",
            "subtitle": "同一开发题 · 模型加载及推理过程的内存观测",
            "cpu8": "CPU · 8 线程 · 进程 RSS",
            "cpu16": "CPU · 16 线程 · 进程 RSS",
            "gpu": "RTX 5090 · 框架分配显存",
            "foot1": "CPU：进程峰值常驻内存；GPU：PyTorch 峰值显存分配。条长对应 MiB。",
            "foot2": "两者计量对象不同，只展示各部署路径的容量需求，不能当作同口径内存节省率。",
            "source": "数据：docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "MiB",
        }
    if language == "en" and kind == "speed":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL SINGLE-QUESTION CASE",
            "title": "Inference latency: shorter is faster",
            "subtitle": "One development request · 703 input tokens · 77 options · model load excluded",
            "cpu8": "CPU · 8 threads · I2_S",
            "cpu16": "CPU · 16 threads · I2_S",
            "gpu": "RTX 5090 · FP16 mixed precision",
            "foot1": "3 repeats for each CPU path; 5 GPU repeats after warmup. Bar length is mean milliseconds.",
            "foot2": "CPU and GPU formats and precision differ. This deployment case is not isolated hardware speedup.",
            "source": "Data: docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "ms",
        }
    if language == "en" and kind == "memory":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL SINGLE-QUESTION CASE",
            "title": "Peak memory: shorter is smaller",
            "subtitle": "One development request · memory observed during model load and inference",
            "cpu8": "CPU · 8 threads · process RSS",
            "cpu16": "CPU · 16 threads · process RSS",
            "gpu": "RTX 5090 · framework VRAM allocation",
            "foot1": "CPU bars are peak process resident memory; GPU bar is peak PyTorch VRAM allocation. Units: MiB.",
            "foot2": "The measures differ. This displays deployment capacity, not a like-for-like memory saving rate.",
            "source": "Data: docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "MiB",
        }
    raise ValueError(f"不支持的语言或图表类型：{language}/{kind}")


def render(language: str, kind: str, values: list[float]) -> None:
    """用同一起点、浅色轨道和数字标签绘制三条可直接比较的数值条。"""
    # 所有条都按原始实测值线性缩放；GPU 延迟短条仍保留最低可见宽度。
    labels = copy_for(language, kind)
    row_names = [labels["cpu8"], labels["cpu16"], labels["gpu"]]
    colors = [CPU_BLUE, CPU_TEAL, GPU_CORAL]
    maximum = max(values)
    figure, axis = plt.subplots(figsize=(14, 7.2), dpi=180, facecolor=PAPER)
    figure.subplots_adjust(left=0.075, right=0.94, top=0.70, bottom=0.23)
    axis.set_xlim(0, maximum * 1.12)
    axis.set_ylim(-0.25, 3.05)
    axis.axis("off")

    # 标题和副标题保持左对齐，阅读顺序与用户提供的横条图一致。
    figure.text(0.075, 0.925, labels["eyebrow"], color=CPU_BLUE, fontsize=11, weight="bold")
    figure.text(0.075, 0.85, labels["title"], color=INK, fontsize=26, weight="bold")
    figure.text(0.075, 0.79, labels["subtitle"], color=MUTED, fontsize=11)

    # 每一行先画等长的细轨道，再按真实值画彩色圆角条。
    for index, (name, value, color) in enumerate(zip(row_names, values, colors)):
        position = 2.5 - index
        axis.text(0, position + 0.28, name, color=INK, fontsize=15, va="bottom")
        axis.text(maximum * 1.12, position + 0.28, f"{value:,.1f} {labels['unit']}",
                  color=INK, fontsize=17, weight="bold", ha="right", va="bottom")
        axis.plot([0, maximum], [position, position], color=TRACK, linewidth=2, zorder=1)
        # 小数值仍显示为一个可见短条，但数字保留原始值，避免视觉零值。
        visible_width = max(value, maximum * 0.018)
        axis.add_patch(FancyBboxPatch((0, position - 0.105), visible_width, 0.21,
                                      boxstyle="round,pad=0,rounding_size=0.105",
                                      linewidth=0, facecolor=color, zorder=2))

    # 数据范围与内存口径写在图内，截图传播时不丢失边界条件。
    figure.text(0.075, 0.135, labels["foot1"], color=INK, fontsize=10)
    figure.text(0.075, 0.092, labels["foot2"], color=MUTED, fontsize=9.5)
    figure.text(0.075, 0.045, labels["source"], color=MUTED, fontsize=8.5)

    # SVG 用于 GitHub 清晰展示，PNG 供 Hugging Face 模型卡引用。
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = FIGURE_DIR / f"bit-jev-case-{kind}.{language}.{extension}"
        figure.savefig(destination, dpi=180, facecolor=PAPER)
        if extension == "svg":
            content = destination.read_text(encoding="utf-8")
            destination.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                                   encoding="utf-8")
    plt.close(figure)


def main() -> None:
    """从已校验的公开 JSON 生成中英两种语言的延迟与内存图。"""
    configure_fonts()
    record = read_and_validate()
    measurements = record["measurements"]
    speed_values = [mean(row["inference_ms"]) for row in measurements]
    memory_values = [measurements[0]["peak_process_rss_mib"],
                     measurements[1]["peak_process_rss_mib"],
                     measurements[2]["peak_gpu_allocation_mib"]]
    for language in ("zh-CN", "en"):
        render(language, "speed", speed_values)
        render(language, "memory", memory_values)
    print("已生成 bit-jev AutoDL 案例的中英文横条对比图")


if __name__ == "__main__":
    main()
