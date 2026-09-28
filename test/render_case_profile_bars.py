"""把公开 AutoDL 单题案例画成适合 README 阅读的横向对比条。"""

from __future__ import annotations

import json
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
EPYC_DATA_PATH = ROOT / "docs" / "benchmark-data" / "bit-jev-epyc9654-cpu32-2026-09-28.json"
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
            "subtitle": "同一编码开发题 · 703 输入 tokens · 77 个候选项 · 不含模型加载",
            "cpu32": "EPYC 9654 · 32 核配额 / 32 线程 · I2_S",
            "gpu": "另一台机器 RTX 5090 · FP16 混合精度",
            "foot1": "EPYC CPU 分两轮重复 6 次；RTX 5090 GPU 重复 5 次，另排除预热。条长为均值。",
            "foot2": "不同机器、模型格式与精度；仅展示部署路径耗时，不能解释为纯硬件加速倍数。",
            "source": "数据：docs/benchmark-data/bit-jev-epyc9654-cpu32-2026-09-28.json + 原 AutoDL 案例",
            "unit": "ms",
        }
    if language == "zh-CN" and kind == "memory":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL 单题案例",
            "title": "峰值占用：越短越省",
            "subtitle": "历史同机案例：Xeon Gold 6459C / RTX 5090 · 加载及推理过程",
            "cpu8": "Xeon CPU · 8 线程 · 进程 RSS",
            "cpu16": "Xeon CPU · 16 线程 · 进程 RSS",
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
            "subtitle": "Same encoded development request · 703 input tokens · 77 options · load excluded",
            "cpu32": "EPYC 9654 · 32-core quota / 32 threads · I2_S",
            "gpu": "RTX 5090 on another host · FP16 mixed precision",
            "foot1": "EPYC CPU: 6 repeats in 2 launches; RTX 5090 GPU: 5 repeats after warmup. Bars show means.",
            "foot2": "Hosts, formats and precision differ. This compares deployment paths, not isolated hardware speedup.",
            "source": "Data: docs/benchmark-data/bit-jev-epyc9654-cpu32-2026-09-28.json + prior AutoDL case",
            "unit": "ms",
        }
    if language == "en" and kind == "memory":
        return {
            "eyebrow": "BIT-JEV  /  AUTODL SINGLE-QUESTION CASE",
            "title": "Peak memory: shorter is smaller",
            "subtitle": "Historical same-host case: Xeon Gold 6459C / RTX 5090 · load and inference",
            "cpu8": "Xeon CPU · 8 threads · process RSS",
            "cpu16": "Xeon CPU · 16 threads · process RSS",
            "gpu": "RTX 5090 · framework VRAM allocation",
            "foot1": "CPU bars are peak process resident memory; GPU bar is peak PyTorch VRAM allocation. Units: MiB.",
            "foot2": "The measures differ. This displays deployment capacity, not a like-for-like memory saving rate.",
            "source": "Data: docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json",
            "unit": "MiB",
        }
    raise ValueError(f"不支持的语言或图表类型：{language}/{kind}")


def read_epyc_and_validate(historical: dict) -> dict:
    """读取新 CPU 测量并核对模型、输入、核心配额与逐次样本。"""
    # 新图只使用公开的脱敏数字，编码请求文件和模型预测不会写进图片。
    record = json.loads(EPYC_DATA_PATH.read_text(encoding="utf-8"))
    workload = record["workload"]
    cpu_measurement = record["measurement"]
    samples = [value for launch in cpu_measurement["inference_ms_by_launch"] for value in launch]
    if record["publication_status"] != "public_sanitized_case_study":
        raise ValueError("EPYC 公开状态不匹配")
    if record["checkpoint"]["cpu_gguf_sha256"] != historical["checkpoint"]["cpu_gguf_sha256"]:
        raise ValueError("EPYC 与历史案例的 GGUF 模型哈希不一致")
    if record["checkpoint"]["head_f32_sha256"] != historical["checkpoint"]["head_f32_sha256"]:
        raise ValueError("EPYC 与历史案例的指针头哈希不一致")
    if (workload["questions"], workload["input_tokens"], workload["options"]) != (1, 703, 77):
        raise ValueError("EPYC 编码请求形状不匹配")
    if workload["input_contents_included"] or len(workload["encoded_input_sha256"]) != 64:
        raise ValueError("EPYC 输入公开边界或摘要格式不匹配")
    if (record["hardware"]["container_cpu_quota_cores"],
            record["timing_protocol"]["native_runner_threads"]) != (32, 32):
        raise ValueError("EPYC CPU 核配额或线程数不匹配")
    if (record["timing_protocol"]["batch"], cpu_measurement["path"]) != (
            128, "native_cpu_i2_s_32_threads"):
        raise ValueError("EPYC 批次或原生路径不匹配")
    if len(samples) != 6 or any(value <= 0 for value in samples):
        raise ValueError("EPYC 32 线程逐次样本无效")
    return record


def render(language: str, kind: str, values: list[float], row_keys: list[str]) -> None:
    """用同一起点、浅色轨道和数字标签绘制两条或三条测量值。"""
    # 所有条都按原始实测值线性缩放；GPU 延迟短条仍保留最低可见宽度。
    labels = copy_for(language, kind)
    row_names = [labels[key] for key in row_keys]
    colors = [CPU_TEAL if key == "cpu16" else CPU_BLUE if key.startswith("cpu") else GPU_CORAL
              for key in row_keys]
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
        position = (2.1 - index * 1.3) if len(values) == 2 else (2.5 - index)
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
    historical = read_and_validate()
    epyc = read_epyc_and_validate(historical)
    measurements = historical["measurements"]
    # 速度图使用新 EPYC 32 线程和历史 RTX 5090；内存图保持原同机案例。
    cpu_samples = [value for launch in epyc["measurement"]["inference_ms_by_launch"] for value in launch]
    speed_values = [mean(cpu_samples), mean(measurements[2]["inference_ms"])]
    memory_values = [measurements[0]["peak_process_rss_mib"],
                     measurements[1]["peak_process_rss_mib"],
                     measurements[2]["peak_gpu_allocation_mib"]]
    for language in ("zh-CN", "en"):
        render(language, "speed", speed_values, ["cpu32", "gpu"])
        render(language, "memory", memory_values, ["cpu8", "cpu16", "gpu"])
    print("已生成 bit-jev AutoDL 案例的中英文横条对比图")


if __name__ == "__main__":
    main()
