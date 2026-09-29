"""创建与公开 Space 同源的本地 Gradio 三题型页面。"""

from __future__ import annotations

import atexit
from functools import partial
from pathlib import Path

import gradio as gr

from .forms import add_form
from .service import ModelSession, WebConfig


# 内容单列排布；说明输入展开后不在狭窄侧栏中逐字换行。
PAGE_CSS = """
.gradio-container { max-width: 1020px !important; margin: 0 auto !important; }
.decision-form { padding: 18px !important; }
.decision-summary { margin-top: 14px; font-size: 1.05rem; }
.jev-results { padding: 18px 20px; border: 1px solid #dce5f2;
  border-radius: 16px; background: #f8fbff; }
.jev-results h3 { margin: 0 0 14px; font-size: 1rem; color: #253858; }
.jev-prob-row + .jev-prob-row { margin-top: 16px; }
.jev-prob-heading { display: flex; justify-content: space-between; gap: 12px;
  margin-bottom: 7px; color: #273951; overflow-wrap: anywhere; }
.jev-prob-heading strong { white-space: nowrap; }
.jev-prob-track { height: 12px; border-radius: 999px; background: #e5ebf5; overflow: hidden; }
.jev-prob-fill { height: 100%; border-radius: inherit; background: #5865e8; }
@media (max-width: 640px) { .decision-form { padding: 10px !important; }
  .jev-results { padding: 14px; } }
"""


def switch_language(language: str) -> list[dict]:
    """同步 Choice、Noul、Score 的中英文面板显隐。"""
    return [gr.update(visible=language == panel_language)
            for _kind in ("choice", "noul", "score")
            for panel_language in ("zh", "en")]


def create_demo(config: WebConfig) -> tuple[gr.Blocks, ModelSession]:
    """创建页面但不下载模型，返回页面与常驻会话。"""
    session = ModelSession(config)
    # Logo 随 wheel 打包，不依赖国内或海外模型站点的图片响应。
    logo = Path(__file__).with_name("logo.png")
    with gr.Blocks(title="bit-jev · JEV Demo", theme=gr.themes.Soft(), css=PAGE_CSS) as demo:
        gr.Image(value=str(logo), show_label=False, height=140,
                 interactive=False, container=False)
        gr.Markdown("# bit-jev\nBitNet · LoRA · teacher–student distillation · I2_S GGUF")
        language = gr.Radio([("中文", "zh"), ("English", "en")], value="zh",
                            label="语言 / Language")
        panels: list[gr.Group] = []
        with gr.Tabs():
            for kind, title in (("choice", "Choice · 选择题"),
                                ("noul", "Noul · 是非题"),
                                ("score", "Score · 等级题")):
                with gr.Tab(title):
                    for panel_language in ("zh", "en"):
                        panels.append(add_form(kind, panel_language,
                                               partial(session.infer, kind, panel_language)))
        language.change(switch_language, inputs=[language], outputs=panels, queue=False)
        if config.public_space:
            note = ("**说明 / Note:** 免费空间使用 2 vCPU 进行推理。 / "
                    "The free Space uses 2 vCPUs for inference.")
        else:
            note = ("**说明 / Note:** 页面在本机运行；首次提交时下载并加载模型。"
                    " / This page runs locally; the first submission downloads and loads the model.")
        gr.Markdown(
            f"{note}\n\n[GitHub](https://github.com/Zeaulo/bit-jev) · "
            "[Hugging Face model](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) · "
            "[ModelScope model](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled)"
        )
    return demo, session


def launch_local(config: WebConfig, host: str = "127.0.0.1", port: int = 7860,
                 open_browser: bool = True) -> None:
    """在指定本机地址启动 Gradio；Ctrl+C 时释放模型资源。"""
    if not 1 <= port <= 65535:
        raise ValueError("端口必须位于 1 至 65535")
    demo, session = create_demo(config)
    atexit.register(session.close)
    # 绑定回环地址，默认仅允许当前电脑访问网页。
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, inbrowser=open_browser,
    )
