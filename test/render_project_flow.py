"""按同一版式生成可在 GitHub 和 Hugging Face 阅读的中英文项目流程图。"""

from __future__ import annotations

from html import escape
from pathlib import Path

import cairosvg


# 所有产物放在文档图目录，运行脚本时不依赖当前工作目录。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

# 两种语言共用节点位置，仅替换精确文案，避免两张图传达不同的模型逻辑。
COPY = {
    "zh-CN": {
        "eyebrow": "BITNET × STRUCTURED DECISIONS",
        "title": "从三值骨干，到直接决策。",
        "subtitle": "训练产物是 I2_S GGUF 与指针头；推理返回候选项分数，不逐 token 生成答案。",
        "train_label": "01  训练与蒸馏",
        "train": [
            ("BitNet BF16", "基础骨干"),
            ("LoRA + 指针头", "任务微调"),
            ("Kev 9B logits", "教师监督"),
            ("学生蒸馏", "更新完整骨干"),
            ("I2_S + head.f32", "量化导出"),
        ],
        "infer_label": "02  安装与推理",
        "infer": [
            ("pip install bit-jev", "安装 Python 包"),
            ("CPU / Vulkan", "Windows x64 AVX2 预编译"),
            ("首次加载模型", "约 1.19 GB，之后复用缓存"),
            ("候选项评分", "答案、概率与原生耗时"),
        ],
        "chip_1": "量化 BitLinear 权重：−1 / 0 / +1",
        "chip_2": "指针头直接读出候选项分数",
        "footer": "多题原生推理逐题运行因果行；速度受输入长度、候选数与硬件影响。",
    },
    "en": {
        "eyebrow": "BITNET × STRUCTURED DECISIONS",
        "title": "From ternary weights to direct decisions.",
        "subtitle": "Training exports I2_S GGUF and a pointer head; inference scores options without answer-token generation.",
        "train_label": "01  TRAIN & DISTILL",
        "train": [
            ("BitNet BF16", "base backbone"),
            ("LoRA + pointer", "task fine-tuning"),
            ("Kev 9B logits", "teacher targets"),
            ("Student distill", "update full backbone"),
            ("I2_S + head.f32", "quantized export"),
        ],
        "infer_label": "02  INSTALL & INFER",
        "infer": [
            ("pip install bit-jev", "install Python package"),
            ("CPU / Vulkan", "prebuilt on Windows x64 AVX2"),
            ("First model load", "~1.19 GB; then cached"),
            ("Score options", "answers, probabilities, latency"),
        ],
        "chip_1": "Quantized BitLinear weights: −1 / 0 / +1",
        "chip_2": "Pointer head reads out option scores",
        "footer": "Native multi-question inference runs one causal row per question; speed depends on request and hardware.",
    },
}


def label(x: int, y: int, value: str, *, size: int = 24, weight: int = 500,
          color: str = "#252a32") -> str:
    """在固定坐标输出转义后的 SVG 文字，防止特殊字符破坏文档。"""
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}">{escape(value)}</text>')


def card(x: int, y: int, width: int, number: str, title: str,
         description: str, *, accent: str) -> str:
    """用统一卡片语法绘制训练或推理节点。"""
    return "".join((
        f'<rect x="{x}" y="{y}" width="{width}" height="146" rx="18" fill="#fff" stroke="#dfe3e7"/>',
        f'<rect x="{x + 20}" y="{y + 20}" width="32" height="32" rx="9" fill="{accent}"/>',
        label(x + 29, y + 43, number, size=15, weight=700, color="#fff"),
        label(x + 20, y + 83, title, size=23, weight=700),
        label(x + 20, y + 117, description, size=16, color="#646d78"),
    ))


def render(language: str) -> None:
    """生成一张 SVG 与对应 PNG，确保 Hub 模型卡不依赖 SVG 渲染支持。"""
    # SVG 直接在 GitHub 渲染；PNG 由同一向量源导出并上传至模型仓库。
    copy = COPY[language]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="880" viewBox="0 0 1600 880">',
        '<rect width="1600" height="880" fill="#f7f8f8"/>',
        '<rect x="36" y="36" width="1528" height="808" rx="28" fill="#fff" stroke="#e1e4e6"/>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">',
        '<circle cx="87" cy="87" r="18" fill="#3358d4"/>',
        '<circle cx="87" cy="87" r="7" fill="#fff"/>',
        label(121, 94, "bit-jev", size=29, weight=800),
        label(1255, 92, copy["eyebrow"], size=13, weight=700, color="#79828d"),
        label(76, 166, copy["title"], size=47, weight=760),
        label(78, 211, copy["subtitle"], size=19, color="#5c6671"),
        '<line x1="76" y1="245" x2="1524" y2="245" stroke="#e8eaec"/>',
        label(77, 294, copy["train_label"], size=21, weight=740, color="#b64f41"),
    ]
    # 训练支线按真实执行顺序排布，节点间箭头仅表示数据依赖。
    for index, (title, description) in enumerate(copy["train"]):
        x = 76 + index * 292
        parts.append(card(x, 317, 266, f"{index + 1:02d}", title, description, accent="#d16856"))
        if index < len(copy["train"]) - 1:
            parts.append(f'<path d="M{x + 271} 390 h13 m-5 -5 5 5 -5 5" fill="none" stroke="#b7bec5" stroke-width="2"/>')
    parts.extend((
        '<path d="M1400 468 v29 H202 v34" fill="none" stroke="#c9cfd5" stroke-width="2" stroke-dasharray="7 7"/>',
        label(77, 554, copy["infer_label"], size=21, weight=740, color="#3358d4"),
    ))
    # 推理支线区分安装、模型缓存、设备选择与直接读出；模型不会在 pip 安装时下载。
    for index, (title, description) in enumerate(copy["infer"]):
        x = 76 + index * 365
        parts.append(card(x, 577, 338, f"{index + 1:02d}", title, description, accent="#3358d4"))
        if index < len(copy["infer"]) - 1:
            parts.append(f'<path d="M{x + 343} 650 h14 m-5 -5 5 5 -5 5" fill="none" stroke="#b7bec5" stroke-width="2"/>')
    parts.extend((
        '<rect x="76" y="753" width="615" height="40" rx="20" fill="#f0f3ff"/>',
        label(94, 780, copy["chip_1"], size=18, weight=650, color="#3358d4"),
        '<rect x="709" y="753" width="815" height="40" rx="20" fill="#fff2ef"/>',
        label(727, 780, copy["chip_2"], size=18, weight=650, color="#b64f41"),
        label(78, 823, copy["footer"], size=15, color="#747d87"),
        '</g></svg>',
    ))
    svg_path = OUTPUT_DIR / f"project-flow.{language}.svg"
    png_path = OUTPUT_DIR / f"project-flow.{language}.png"
    svg_path.write_text("".join(parts), encoding="utf-8")
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    print(f"生成 {svg_path} 与 {png_path}")


def main() -> None:
    """为中文首页和英文首页生成内容对应的两套图。"""
    # 两种语言使用同一节点图结构，避免手工维护时出现步骤漂移。
    for language in COPY:
        render(language)


if __name__ == "__main__":
    main()
