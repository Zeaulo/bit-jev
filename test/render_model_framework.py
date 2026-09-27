"""依据 bit-jev 的实际前向路径绘制可复现的深度学习模型结构图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# 图形只写入公开文档目录；运行脚本始终位于项目根目录的 test 下。
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

# 一组颜色对应编码、骨干、读出和量化部署四种职责。
INK = "#18243B"
MUTED = "#53647A"
LINE = "#D8E2EF"
PAPER = "#F5F8FC"
WHITE = "#FFFFFF"
BLUE = "#186CE5"
BLUE_PALE = "#EAF2FF"
PURPLE = "#7052C7"
PURPLE_PALE = "#F0EBFC"
TEAL = "#008D83"
TEAL_PALE = "#E3F6F3"
ORANGE = "#B46913"
ORANGE_PALE = "#FFF2DE"


def box(axis, x, y, w, h, color=WHITE, edge=LINE, radius=0.012, width=1.1):
    """画一块有明确坐标的模块面板，供连接线复用边界。"""
    panel = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=width, edgecolor=edge, facecolor=color,
        transform=axis.transAxes, zorder=2,
    )
    axis.add_patch(panel)
    return panel


def label(axis, x, y, text, size=9, color=INK, weight="normal",
          align="left", vertical="center"):
    """按画布坐标写字，保持所有模块字号和对齐方式一致。"""
    axis.text(x, y, text, transform=axis.transAxes, fontsize=size, color=color,
              fontweight=weight, ha=align, va=vertical, zorder=5)


def arrow(axis, start, end, color=BLUE, width=1.6, style="-|>"):
    """沿数据流方向画箭头，不用箭头暗示尚未实现的并行执行。"""
    axis.add_patch(FancyArrowPatch(
        start, end, transform=axis.transAxes,
        arrowstyle=style, mutation_scale=11, linewidth=width,
        color=color, shrinkA=0, shrinkB=0, zorder=4,
    ))


def step(axis, number, title, y):
    """给每一层数据流统一编号和章节标题。"""
    box(axis, 0.035, y - 0.014, 0.038, 0.033, BLUE, BLUE, radius=0.008)
    label(axis, 0.054, y + 0.002, number, 9, WHITE, "bold", "center")
    label(axis, 0.085, y + 0.002, title, 13, INK, "bold")


def mask(axis, x, y, cell=0.020):
    """按代码中的 j≤i 且同分支或共享 state 规则画出 6×6 掩码。"""
    segments = [0, 0, 1, 1, 2, 2]
    names = ["S", "S", "Q1", "Q1", "Q2", "Q2"]
    for row, query_segment in enumerate(segments):
        for column, key_segment in enumerate(segments):
            allowed = column <= row and (key_segment == 0 or key_segment == query_segment)
            axis.add_patch(Rectangle(
                (x + column * cell, y + (5 - row) * cell), cell * 0.94, cell * 0.94,
                facecolor=BLUE if allowed else WHITE, edgecolor=LINE, linewidth=0.65,
                transform=axis.transAxes, zorder=3,
            ))
        label(axis, x - 0.008, y + (5 - row) * cell + cell / 2,
              names[row], 6.8, MUTED, align="right")
    for column, name in enumerate(names):
        label(axis, x + column * cell + cell / 2, y + 6 * cell + 0.009,
              name, 6.8, MUTED, align="center")
    label(axis, x, y - 0.018, "query rows  /  key columns", 7, MUTED)


def input_row(axis):
    """展示输入封装与注意力隔离的精确关系。"""
    step(axis, "01", "INPUT PACKING + BRANCH ISOLATION", 0.866)
    box(axis, 0.035, 0.684, 0.93, 0.154)

    box(axis, 0.052, 0.710, 0.185, 0.094, BLUE_PALE, radius=0.010)
    label(axis, 0.065, 0.781, "One request", 10, BLUE, "bold")
    label(axis, 0.065, 0.753, "shared state + Q1, Q2", 8.5)
    label(axis, 0.065, 0.728, "caller-supplied options", 8.5)

    arrow(axis, (0.244, 0.757), (0.270, 0.757))
    box(axis, 0.272, 0.710, 0.400, 0.094, WHITE, LINE, radius=0.010)
    label(axis, 0.286, 0.781, "Reserved delimiters + token / segment / position IDs",
          9.3, INK, "bold")
    label(axis, 0.286, 0.754, "[ state ]   [ Q1: opt1 | opt2 | decide ]", 8.6, MUTED)
    label(axis, 0.286, 0.730, "                    [ Q2: opt1 | opt2 | decide ]", 8.6, MUTED)

    arrow(axis, (0.678, 0.757), (0.706, 0.757))
    box(axis, 0.708, 0.701, 0.239, 0.116, BLUE_PALE, radius=0.010)
    label(axis, 0.720, 0.798, "Branch mask", 9.2, BLUE, "bold")
    mask(axis, 0.825, 0.719, cell=0.013)
    label(axis, 0.720, 0.762, "Shared state visible", 7.7)
    label(axis, 0.720, 0.738, "Q1 ↮ Q2", 7.7)


def decoder_layer(axis):
    """展开一个 BitNet 层，标注四处归一化和两条残差链路。"""
    box(axis, 0.261, 0.367, 0.555, 0.254, WHITE, BLUE, radius=0.011, width=1.5)
    label(axis, 0.279, 0.600, "BitNet decoder layer  × 30", 12, BLUE, "bold")
    label(axis, 0.799, 0.600, "d = 2560", 8.5, MUTED, align="right")

    # 注意力与前馈模块按 native/llama.cpp 的真实执行顺序从左向右排列。
    boxes = [
        (0.280, 0.467, 0.085, 0.083, BLUE_PALE, "RMSNorm", "attn norm"),
        (0.388, 0.467, 0.151, 0.083, BLUE_PALE, "Q / K / V", "RoPE(Q,K) + mask"),
        (0.560, 0.467, 0.113, 0.083, BLUE_PALE, "Attention", "subnorm + O"),
        (0.696, 0.467, 0.101, 0.083, BLUE_PALE, "Residual", "x + attn"),
        (0.280, 0.379, 0.085, 0.065, PURPLE_PALE, "RMSNorm", "ffn norm"),
        (0.388, 0.379, 0.151, 0.065, PURPLE_PALE, "Up × ReLU²(Gate)", "BitLinear"),
        (0.560, 0.379, 0.113, 0.065, PURPLE_PALE, "Subnorm", "Down proj"),
        (0.696, 0.379, 0.101, 0.065, PURPLE_PALE, "Residual", "x + ffn"),
    ]
    for x, y, w, h, fill, title, subtitle in boxes:
        box(axis, x, y, w, h, fill, radius=0.007)
        label(axis, x + w / 2, y + h * 0.65, title, 8.4, INK, "bold", "center")
        label(axis, x + w / 2, y + h * 0.27, subtitle, 7.2, MUTED, align="center")
    for first, second in ((0.365, 0.388), (0.539, 0.560), (0.673, 0.696)):
        arrow(axis, (first, 0.507), (second, 0.507), BLUE, 1.1)
        arrow(axis, (first, 0.411), (second, 0.411), PURPLE, 1.1)
    arrow(axis, (0.746, 0.462), (0.746, 0.445), PURPLE, 1.0)
    arrow(axis, (0.746, 0.445), (0.270, 0.445), PURPLE, 1.0, "-")
    arrow(axis, (0.270, 0.445), (0.270, 0.411), PURPLE, 1.0)
    label(axis, 0.279, 0.572, "Ternary BitLinear projections  {−1, 0, +1}; residual stream stays higher precision",
          8.1, MUTED)


def backbone_row(axis):
    """连接嵌入、30 层 BitNet 骨干与最终隐藏状态。"""
    step(axis, "02", "BITNET b1.58 BACKBONE  ·  NO LM-HEAD DECODING", 0.645)
    box(axis, 0.035, 0.344, 0.93, 0.282)
    box(axis, 0.052, 0.444, 0.172, 0.105, TEAL_PALE, radius=0.010)
    label(axis, 0.138, 0.521, "Token embeddings", 9.2, TEAL, "bold", "center")
    label(axis, 0.138, 0.493, "position IDs", 8.5, INK, align="center")
    label(axis, 0.138, 0.467, "hidden width 2560", 8.2, MUTED, align="center")
    arrow(axis, (0.226, 0.495), (0.260, 0.495))
    decoder_layer(axis)
    arrow(axis, (0.818, 0.495), (0.843, 0.495))
    box(axis, 0.845, 0.444, 0.101, 0.105, TEAL_PALE, radius=0.010)
    label(axis, 0.895, 0.521, "Final", 9, TEAL, "bold", "center")
    label(axis, 0.895, 0.494, "RMSNorm", 8.3, INK, align="center")
    label(axis, 0.895, 0.468, "H ∈ Rᴸˣ²⁵⁶⁰", 8, MUTED, align="center")
    label(axis, 0.050, 0.360, "LoRA may adapt Q/K/V/O and Gate/Up/Down during training; inference reads hidden states, not vocabulary logits.",
          8.1, MUTED)


def pointer_row(axis):
    """展开两个线性投影、候选点积、温度和结构化输出。"""
    step(axis, "03", "POINTER READOUT  ·  ONE SCORE PER SUPPLIED OPTION", 0.317)
    box(axis, 0.035, 0.091, 0.93, 0.201)

    # 决策位与各候选结束位来自同一前向隐藏状态张量。
    box(axis, 0.054, 0.184, 0.159, 0.077, TEAL_PALE, radius=0.010)
    label(axis, 0.133, 0.237, "h[decide]", 10, TEAL, "bold", "center")
    label(axis, 0.133, 0.207, "one per question", 8, MUTED, align="center")
    box(axis, 0.054, 0.102, 0.159, 0.069, TEAL_PALE, radius=0.010)
    label(axis, 0.133, 0.148, "h[option-close]ᵢ", 9.5, TEAL, "bold", "center")
    label(axis, 0.133, 0.121, "i = 1 … K", 8, MUTED, align="center")

    box(axis, 0.258, 0.187, 0.137, 0.070, PURPLE_PALE, radius=0.009)
    label(axis, 0.326, 0.231, "q = Wq h + bq", 9, PURPLE, "bold", "center")
    label(axis, 0.326, 0.205, "256 dimensions", 8, MUTED, align="center")
    box(axis, 0.258, 0.103, 0.137, 0.067, PURPLE_PALE, radius=0.009)
    label(axis, 0.326, 0.147, "kᵢ = Wk hᵢ + bk", 8.8, PURPLE, "bold", "center")
    label(axis, 0.326, 0.121, "K × 256", 8, MUTED, align="center")
    arrow(axis, (0.216, 0.223), (0.256, 0.223), PURPLE)
    arrow(axis, (0.216, 0.137), (0.256, 0.137), PURPLE)

    box(axis, 0.454, 0.141, 0.149, 0.088, ORANGE_PALE, radius=0.010)
    label(axis, 0.528, 0.199, "logitᵢ = kᵢ · q / √256", 8.9, ORANGE, "bold", "center")
    label(axis, 0.528, 0.168, "eval: divide by T", 8, MUTED, align="center")
    arrow(axis, (0.398, 0.222), (0.452, 0.192), PURPLE)
    arrow(axis, (0.398, 0.137), (0.452, 0.174), PURPLE)

    box(axis, 0.663, 0.141, 0.126, 0.088, ORANGE_PALE, radius=0.010)
    label(axis, 0.726, 0.199, "softmax", 9.5, ORANGE, "bold", "center")
    label(axis, 0.726, 0.169, "P(optionᵢ)", 8, MUTED, align="center")
    arrow(axis, (0.606, 0.185), (0.661, 0.185), ORANGE)

    box(axis, 0.844, 0.141, 0.102, 0.088, BLUE_PALE, radius=0.010)
    label(axis, 0.895, 0.198, "Typed answer", 9, BLUE, "bold", "center")
    label(axis, 0.895, 0.169, "choice / noul / score", 7.7, MUTED, align="center")
    arrow(axis, (0.792, 0.185), (0.842, 0.185), BLUE)


def footer(axis):
    """给打包训练路径和原生 CPU 路径添加准确的执行边界。"""
    box(axis, 0.035, 0.022, 0.447, 0.050, BLUE_PALE, radius=0.008)
    label(axis, 0.050, 0.055, "PACKED PYTORCH", 8.5, BLUE, "bold")
    label(axis, 0.050, 0.036, "branch mask shares state work across questions", 7.9, MUTED)
    box(axis, 0.500, 0.022, 0.465, 0.050, PURPLE_PALE, radius=0.008)
    label(axis, 0.515, 0.055, "NATIVE I2_S CPU", 8.5, PURPLE, "bold")
    label(axis, 0.515, 0.036, "one causal row per question; shared state is recomputed", 7.9, MUTED)


def main():
    """按真实计算顺序绘图并输出可缩放 SVG 与高清 PNG。"""
    figure = plt.figure(figsize=(16, 10), dpi=180, facecolor=PAPER)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    label(axis, 0.035, 0.961, "bit-jev  |  neural model framework",
          22, INK, "bold")
    label(axis, 0.035, 0.926,
          "BitNet backbone + Kev-style pointer head  ·  structured decisions without answer-token generation",
          11, MUTED)
    input_row(axis)
    arrow(axis, (0.500, 0.684), (0.500, 0.627), BLUE, 1.8)
    backbone_row(axis)
    arrow(axis, (0.500, 0.344), (0.500, 0.293), TEAL, 1.8)
    pointer_row(axis)
    footer(axis)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = FIGURES / f"model-framework.{extension}"
        figure.savefig(destination, dpi=180, facecolor=PAPER)
        if extension == "svg":
            # Matplotlib 的 SVG 路径行尾带空格，写入 Git 前清理它们。
            lines = destination.read_text(encoding="utf-8").splitlines()
            destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)
    print(f"模型结构图：{FIGURES / 'model-framework.svg'}")


if __name__ == "__main__":
    main()
