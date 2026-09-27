"""为中英文首页生成同版式的 BitNet + bit-jev 核心卖点图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# 生成程序位于 test，图片集中保存在文档目录以便 README 直接引用。
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
)

# 蓝色标识量化，青色标识 CPU 部署，紫色标识训练与蒸馏。
PAPER = "#F4F7FC"
WHITE = "#FFFFFF"
INK = "#17233A"
MUTED = "#50617A"
HAIRLINE = "#DCE5F0"
BLUE = "#176DE8"
BLUE_SOFT = "#EAF2FF"
TEAL = "#008F86"
TEAL_SOFT = "#E2F6F3"
PURPLE = "#7452CE"
PURPLE_SOFT = "#F0EBFD"
ORANGE = "#BD7115"
ORANGE_SOFT = "#FFF1DC"


# 所有公开数字都来自微软原版 BitNet GGUF 的独立实测，不属于 bit-jev 检查点。
CONTENT = {
    "zh-CN": {
        "title": "bit-jev：三值 BitNet × 结构化判断",
        "subtitle": "核心链路：LoRA 微调 → 教师–学生蒸馏 → I2_S 量化 → 原生 CPU 选项评分",
        "ternary_title": "01  三值权重",
        "ternary_main": "−1       0       +1",
        "ternary_line1": "BitLinear 权重在量化推理时取三值",
        "ternary_line2": "指针头与部分张量仍为较高精度",
        "memory_title": "02  低内存 · CPU 友好",
        "memory_main": "1.20 GiB",
        "memory_line1": "CPU 进程峰值 RSS",
        "memory_line2": "同一公开 GGUF 文件：1.72 GiB",
        "speed_title": "03  CPU 速度实测",
        "speed_main": "56.23 token/s",
        "speed_line1": "128 输入 token 预填充中位吞吐",
        "speed_line2": "32 输出 token 解码：4.57 token/s",
        "training_title": "04  从适配到蒸馏，再到 CPU 部署",
        "steps": (
            ("BitNet BF16 骨干", "公开基础模型"),
            ("LoRA 微调", "训练指针头"),
            ("教师输出", "选项 logits"),
            ("学生蒸馏", "全骨干训练"),
            ("I2_S + CPU", "候选项打分"),
        ),
        "flow_note": "学生从基础骨干初始化；指针头来自微调模型。原生 CPU 路径当前逐题计算。",
        "evidence": "右侧内存与速度仅来自微软公开 BitNet 原版 I2_S GGUF · Ryzen 7 4800H / 8 线程 · 每阶段 5 次 · 吞吐不计模型加载",
        "boundary": "尚未公开 bit-jev 训练权重及其端到端速度；预填充 token/s 不等于结构化请求吞吐。",
        "badge": "性能数字仅来自公开基础模型",
    },
    "en": {
        "title": "bit-jev: ternary BitNet × structured decisions",
        "subtitle": "LoRA fine-tune → teacher–student distillation → I2_S quantization → native CPU option scoring",
        "ternary_title": "01  Ternary weights",
        "ternary_main": "−1       0       +1",
        "ternary_line1": "Quantized BitLinear weights take three values",
        "ternary_line2": "Pointer head and some tensors keep higher precision",
        "memory_title": "02  Low memory · CPU friendly",
        "memory_main": "1.20 GiB",
        "memory_line1": "Peak CPU process RSS",
        "memory_line2": "Same public GGUF file: 1.72 GiB",
        "speed_title": "03  Measured CPU throughput",
        "speed_main": "56.23 tok/s",
        "speed_line1": "Median prefill, 128 input tokens",
        "speed_line2": "Decode, 32 output tokens: 4.57 tok/s",
        "training_title": "04  Fine-tune, distill, deploy",
        "steps": (
            ("BitNet BF16 base", "public model"),
            ("LoRA fine-tune", "train pointer head"),
            ("Teacher outputs", "option logits"),
            ("Student distill", "full backbone"),
            ("I2_S + CPU", "score options"),
        ),
        "flow_note": "Student starts from base weights; pointer head comes from the fine-tuned run. Native CPU currently computes one row per question.",
        "evidence": "Memory and speed above: independent Microsoft public BitNet I2_S GGUF test · Ryzen 7 4800H / 8 threads · 5 runs per phase · load excluded from throughput",
        "boundary": "No public bit-jev checkpoint or end-to-end latency yet. Prefill tokens/s is not structured-request throughput.",
        "badge": "PERFORMANCE NUMBERS: PUBLIC BASE ONLY",
    },
}


def configure_font() -> None:
    """优先安装系统中文字体，并保持 SVG 与 PNG 的文字形态一致。"""
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def panel(axis, x, y, width, height, *, fill=WHITE, border=HAIRLINE, radius=0.015):
    """画可复用的信息卡片，坐标采用整张图的比例。"""
    shape = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.1, edgecolor=border, facecolor=fill,
        transform=axis.transAxes, zorder=2,
    )
    axis.add_patch(shape)


def text(axis, x, y, message, size, color=INK, weight="normal", align="left"):
    """统一字号、颜色和锚点，避免两种语言落在不同视觉层级。"""
    axis.text(x, y, message, transform=axis.transAxes,
              fontsize=size, color=color, fontweight=weight,
              ha=align, va="center", zorder=4)


def arrow(axis, left, right):
    """从训练阶段左侧流向下一阶段，显示实现中的先后顺序。"""
    axis.add_patch(FancyArrowPatch(
        left, right, transform=axis.transAxes, arrowstyle="-|>",
        mutation_scale=14, linewidth=1.8, color=PURPLE,
        shrinkA=0, shrinkB=0, zorder=4,
    ))


def feature_card(axis, x, title, main, first, second, *,
                 accent, tint, main_size=24):
    """绘制三值、内存、速度三项可独立阅读的核心事实。"""
    panel(axis, x, 0.485, 0.286, 0.315)
    panel(axis, x + 0.017, 0.747, 0.010, 0.033, fill=accent, border=accent, radius=0.004)
    text(axis, x + 0.039, 0.762, title, 13, accent, "bold")
    panel(axis, x + 0.017, 0.596, 0.252, 0.122, fill=tint, border=tint, radius=0.011)
    text(axis, x + 0.143, 0.656, main, main_size, accent, "bold", "center")
    text(axis, x + 0.018, 0.555, first, 9.8, INK)
    text(axis, x + 0.018, 0.519, second, 9.2, MUTED)


def training_row(axis, strings):
    """按源码的微调、教师导出、全骨干学生和原生导出顺序排布。"""
    panel(axis, 0.045, 0.181, 0.910, 0.252)
    text(axis, 0.065, 0.392, strings["training_title"], 13.5, PURPLE, "bold")
    colors = (BLUE_SOFT, PURPLE_SOFT, ORANGE_SOFT, PURPLE_SOFT, TEAL_SOFT)
    accents = (BLUE, PURPLE, ORANGE, PURPLE, TEAL)
    for index, ((name, detail), color, accent) in enumerate(zip(strings["steps"], colors, accents)):
        x = 0.064 + index * 0.180
        panel(axis, x, 0.259, 0.158, 0.092, fill=color, border=color, radius=0.010)
        text(axis, x + 0.079, 0.318, name, 10.4, accent, "bold", "center")
        text(axis, x + 0.079, 0.283, detail, 8.8, MUTED, align="center")
        if index < 4:
            arrow(axis, (x + 0.160, 0.305), (x + 0.178, 0.305))
    text(axis, 0.065, 0.216, strings["flow_note"], 8.9, MUTED)


def render(language: str) -> None:
    """按语言生成一张独立 SVG 和 PNG，保证版本口径一致。"""
    strings = CONTENT[language]
    figure = plt.figure(figsize=(16, 8.8), dpi=180, facecolor=PAPER)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    text(axis, 0.045, 0.931, strings["title"], 24, INK, "bold")
    text(axis, 0.045, 0.883, strings["subtitle"], 11.2, MUTED)
    panel(axis, 0.682, 0.829, 0.273, 0.037, fill=TEAL_SOFT, border=TEAL_SOFT, radius=0.008)
    text(axis, 0.819, 0.848, strings["badge"], 8.7, TEAL, "bold", "center")

    feature_card(axis, 0.045, strings["ternary_title"], strings["ternary_main"],
                 strings["ternary_line1"], strings["ternary_line2"],
                 accent=BLUE, tint=BLUE_SOFT, main_size=26)
    feature_card(axis, 0.357, strings["memory_title"], strings["memory_main"],
                 strings["memory_line1"], strings["memory_line2"],
                 accent=TEAL, tint=TEAL_SOFT, main_size=29)
    feature_card(axis, 0.669, strings["speed_title"], strings["speed_main"],
                 strings["speed_line1"], strings["speed_line2"],
                 accent=ORANGE, tint=ORANGE_SOFT, main_size=24)
    training_row(axis, strings)

    # 两行证据边界随图嵌入，防止图片脱离 README 后被误认为 bit-jev 跑分。
    text(axis, 0.045, 0.116, strings["evidence"], 8.5, MUTED)
    text(axis, 0.045, 0.076, strings["boundary"], 8.7, MUTED)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = FIGURES / f"model-highlights.{language}.{extension}"
        figure.savefig(destination, dpi=180, facecolor=PAPER)
        if extension == "svg":
            # 清理 Matplotlib 路径数据的尾随空格，供 Git 空白检查。
            lines = destination.read_text(encoding="utf-8").splitlines()
            destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def main() -> None:
    """一次生成中文和英文两份图，不允许图文数字漂移。"""
    configure_font()
    for language in CONTENT:
        render(language)
        print(f"已生成 {language} 图：{FIGURES / f'model-highlights.{language}.svg'}")


if __name__ == "__main__":
    main()
