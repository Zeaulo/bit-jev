"""生成 bit-jev 论文式架构图（matplotlib，输出 docs/figures/architecture.svg 与 .png）。

论文风格：白底、直角细框、左到右数据流；骨干框内给出量化公式与 30 层堆叠，
蓝色支路表示 round(W) 后的 I2_S GGUF 走原生 CPU 推理，底部附 1.58 bit 说明。
运行：python test/render_architecture_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 图中含中文标注，注册 Windows 中文字体（微软雅黑/黑体）避免 PNG 缺字。
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免缺字

INK = "#111111"
MUTED = "#444444"
BLUE = "#1f4e9c"
PAPER = "#ffffff"


def box(ax, x, y, w, h, text_lines, fill=PAPER, dashed=False, title=None, title_dy=None):
    """画一个论文式方框：白底、细黑边、可加粗标题与若干正文行。

    注意 boxstyle 的 pad 会向外扩张，方框视觉边为 (x±pad, y±pad)；
    箭头端点必须取视觉边，否则会埋进框内。
    """
    style = dict(boxstyle="square,pad=0.006", linewidth=1.2,
                 edgecolor=INK, facecolor=fill)
    if dashed:
        style["linestyle"] = (0, (4, 3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes, **style))
    cx = x + w / 2
    if title is not None:
        ax.text(cx, y + (title_dy if title_dy is not None else h - 0.05), title,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
    for i, line in enumerate(text_lines):
        ax.text(cx, y + h - 0.09 - i * 0.045, line, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.6, color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=color, shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(13.2, 7.4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PAPER, edgecolor="none", zorder=-10))

    # 标题
    ax.text(0.5, 0.962, "bit-jev: Structured Decisions Without Answer Decoding",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=15.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.933, "bit-jev = bitnet + jev ｜ BitNet b1.58 backbone (ternary weights) + Kev-style pointer head",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, color=MUTED, style="italic")

    # ---- 输入请求（左） ----
    box(ax, 0.025, 0.30, 0.175, 0.46, [
        "<|state|> state …",
        "",
        "<|q|> 问题 1",
        "  <|opt|> 选项1 </opt>",
        "  …   <|decide|>",
        "",
        "<|q|> 问题 2 …",
        "…",
    ], fill="#f6f6f6", title="请求 Request", title_dy=0.415)

    # ---- 编码 ----
    box(ax, 0.235, 0.40, 0.125, 0.24, [
        "Reserved tokens",
        "分支编号 seg",
        "位置 id 重启",
    ], title="分词与边界", title_dy=0.205)

    # ---- BitNet 骨干（中）----
    box(ax, 0.395, 0.24, 0.20, 0.52, [
        "Embedding",
        "",
        "",
        "",
        "W = round(W·s)·clip(−1,1)/s",
        "s = 1 / mean|W|",
        "W ∈ {−1, 0, +1}·s",
        "激活 8-bit",
    ], fill="#f2f6fd", title="BitNet b1.58 骨干", title_dy=0.475)

    # ---- 骨干内部的 30 层堆叠：四条薄框表示重复的 Transformer 层 ----
    for _i in range(4):
        _by = 0.506 + _i * 0.040                  # 层高 0.028、层间距 0.012
        ax.add_patch(plt.Rectangle((0.415, _by), 0.11, 0.028,
                                   transform=ax.transAxes, linewidth=0.9,
                                   edgecolor=INK, facecolor="#ffffff", zorder=2))
    ax.text(0.535, 0.580, "× 30 层", transform=ax.transAxes, ha="left",
            va="center", fontsize=9, color=MUTED)

    # ---- 分块因果掩码说明（骨干下方，虚线；蓝色支路从其右侧绕行） ----
    box(ax, 0.395, 0.05, 0.155, 0.16, [
        "分支可读 state 与自身；",
        "问题之间互不可见",
    ], dashed=True, title="分块因果掩码（打包路径）", title_dy=0.11)

    # ---- 隐藏状态 ----
    box(ax, 0.63, 0.40, 0.13, 0.24, [
        "h[</opt>]  选项尾",
        "h[<|decide|>] 决策位",
    ], title="隐藏状态", title_dy=0.205)

    # ---- 指针头 ----
    box(ax, 0.795, 0.28, 0.185, 0.48, [
        "q = Wq h[decide]",
        "ki = Wk h[/opt]i",
        "scorei = (q·ki)/√d",
        "softmax → pi",
        "",
        "温度 T 校准",
        "fp32，约 1.3 M 参数",
    ], fill="#f2f6fd", title="指针头 Pointer Head", title_dy=0.435)

    # ---- 输出 ----
    box(ax, 0.795, 0.075, 0.185, 0.155, [
        "choice: argmax + pi",
        "noul: p(yes)   score: E[level]",
    ], fill="#f6f6f6", title="答案 Answers", title_dy=0.125)

    # ---- 主数据流箭头（端点在方框 pad 扩张后的视觉边上，避免埋进框内） ----
    arrow(ax, 0.206, 0.53, 0.229, 0.53)
    arrow(ax, 0.366, 0.53, 0.389, 0.53)
    arrow(ax, 0.601, 0.53, 0.624, 0.53)
    arrow(ax, 0.766, 0.53, 0.789, 0.53)
    arrow(ax, 0.887, 0.274, 0.887, 0.236)

    # ---- 蓝色 CPU 部署支路：从骨干底部右侧下行，经掩码框右侧绕到答案框 ----
    ax.add_patch(FancyArrowPatch((0.57, 0.234), (0.57, 0.055),
                                 transform=ax.transAxes, arrowstyle="-",
                                 linewidth=1.4, color=BLUE))
    ax.add_patch(FancyArrowPatch((0.57, 0.055), (0.855, 0.055),
                                 transform=ax.transAxes, arrowstyle="-",
                                 linewidth=1.4, color=BLUE))
    ax.add_patch(FancyArrowPatch((0.855, 0.055), (0.855, 0.069),
                                 transform=ax.transAxes, arrowstyle="-",
                                 linewidth=1.4, color=BLUE))
    ax.text(0.70, 0.078, "量化骨干 → I2_S GGUF",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.8, color=BLUE, style="italic")
    ax.text(0.70, 0.032, "原生 CPU runner：逐题一条因果行 → h → 指针头",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.8, color=BLUE, style="italic")

    # ---- 底部注释 ----
    ax.text(0.025, 0.042, "打包路径可共享 state；原生 CPU 路径逐题计算；不生成答案 token。",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.8, color=MUTED)
    ax.text(0.025, 0.018, "BitLinear 使用三值权重；模型文件与运行内存需分别实测。",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.8, color=MUTED)

    # ---- 图例（放在副标题与主干框之间的空带，避免与标题同行重叠） ----
    ax.add_patch(plt.Rectangle((0.025, 0.882), 0.016, 0.022, transform=ax.transAxes,
                               facecolor=PAPER, edgecolor=INK, linewidth=1.2))
    ax.text(0.048, 0.893, "模块", transform=ax.transAxes, va="center", fontsize=8.6, color=MUTED)
    ax.plot([0.10, 0.13], [0.893, 0.893], transform=ax.transAxes, color=INK, linewidth=1.3)
    ax.text(0.136, 0.893, "前向数据流", transform=ax.transAxes, va="center", fontsize=8.6, color=MUTED)
    ax.plot([0.21, 0.24], [0.893, 0.893], transform=ax.transAxes, color=BLUE, linewidth=1.4)
    ax.text(0.246, 0.893, "CPU 推理路径（I2_S）", transform=ax.transAxes, va="center", fontsize=8.6, color=BLUE)

    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "architecture"
    svg_path = out.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=PAPER,
                metadata={"Date": None})
    # 清理 Matplotlib 在路径数据行尾写入的空格，保持 SVG 可复现且通过差异检查。
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                        encoding="utf-8")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor=PAPER, dpi=150)
    plt.close(fig)
    print("wrote", out.with_suffix(".svg"), "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
