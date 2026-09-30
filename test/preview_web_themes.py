"""不下载模型，使用正式概率 HTML 验证亮色、暗色和长文本布局。"""

import sys
from pathlib import Path

# 将本工作区源码置于已安装包之前，确保预览本次修复。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import gradio as gr
from bit_jev.web.presentation import render_probabilities
from bit_jev.web.theme import PAGE_CSS


def main() -> None:
    """创建三个题型的中英文概率卡，供浏览器检查主题切换。"""
    # 固定数值仅用于排版验收，不是模型实测输出。
    examples = {
        "choice": {"type": "choice", "probabilities": {
            "比特币：较长的中文选项文字，用于检查换行是否影响百分比": 0.581,
            "黄金 Gold": 0.419}},
        "noul": {"type": "noul", "answer": 0.873},
        "score": {"type": "score", "probabilities": {
            "level: 低 Low": 0.123, "level: 中 Medium": 0.333,
            "level: 高 High": 0.544}},
    }
    with gr.Blocks(theme=gr.themes.Soft(), css=PAGE_CSS) as demo:
        gr.Markdown("# 概率区域主题验收 / Probability theme preview")
        for kind, detail in examples.items():
            # 每行同时展示两种语言，复用正式页面的安全 HTML 渲染函数。
            gr.Markdown(f"## {kind}")
            for language in ("zh", "en"):
                gr.HTML(render_probabilities(detail, language))
    demo.launch(server_name="127.0.0.1", server_port=7863, inbrowser=False)


if __name__ == "__main__":
    main()
