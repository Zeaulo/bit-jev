"""ModelScope Docker Space 的中英双语 CPU 推理入口。"""

from __future__ import annotations

import os
import threading
from functools import partial

import gradio as gr
from bit_jev.gguf import BitJev

from forms import add_form
from logic import build_choice_request, build_noul_request, build_score_request, present_result


# 模型进程常驻且串行接收 JSONL，所有题型共用这一份加载后的权重。
_model: BitJev | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()
LOGO_URL = "https://modelscope.cn/models/JingHao9616/bit-jev-2b-distilled/resolve/master/bit-jev-logo.png"


def get_model() -> BitJev:
    """首次请求按需下载模型，此后复用同一原生进程。"""
    global _model
    with _model_lock:
        if _model is None:
            # Docker 镜像构建时已完成 Linux 原生编译，访客无须安装构建工具。
            _model = BitJev.from_pretrained(
                device="cpu",
                threads=min(os.cpu_count() or 2, 2),
                source=os.environ.get("BIT_JEV_MODEL_SOURCE", "auto"),
                binary=os.environ.get("BIT_JEV_BINARY", "/opt/bit-jev-cpu"),
            )
        return _model


def infer(kind: str, language: str, *values: object) -> tuple[str, dict]:
    """先校验表单，再执行真实推理并呈现可读结果。"""
    try:
        if kind == "noul":
            state, question, no_description, yes_description = values
            request = build_noul_request(state, question, language,
                                         no_description, yes_description)
        else:
            count, state, question, *entries = values
            # 隐藏行仍在 Gradio 输入列表内，按当前可见行数截取即可。
            active_count = int(count)
            options = entries[:4][:active_count]
            descriptions = entries[4:][:active_count]
            request = (build_choice_request(state, question, options, descriptions)
                       if kind == "choice" else
                       build_score_request(state, question, options, descriptions))
        with _infer_lock:
            result = get_model().infer(request)
        return present_result(result, language)
    except Exception as error:
        label = "运行失败" if language == "zh" else "Inference failed"
        return f"{label}: `{type(error).__name__}: {error}`", {}


def switch_language(language: str) -> list[dict]:
    """同步三个题型面板的语言显隐状态。"""
    # 输出顺序与创建时的任务、语言顺序一致。
    return [gr.update(visible=language == panel_language)
            for _kind in ("choice", "noul", "score")
            for panel_language in ("zh", "en")]


with gr.Blocks(title="bit-jev · Live CPU Demo", theme=gr.themes.Soft()) as demo:
    # 模型仓库的公开 PNG 返回真实图片字节，避免 Studio 中 LFS 指针被当作图片。
    gr.HTML(f'<img src="{LOGO_URL}" alt="bit-jev 0/1 rocket logo" '
            'style="display:block;width:140px;max-width:36vw;margin:0 auto" />')
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
                                           partial(infer, kind, panel_language)))
    language.change(switch_language, inputs=[language], outputs=panels, queue=False)
    gr.Markdown(
        "**说明 / Note:** 免费空间使用 2 vCPU 进行推理。 / "
        "The free Space uses 2 vCPUs for inference.\n\n"
        "[GitHub](https://github.com/Zeaulo/bit-jev) · "
        "[Hugging Face model](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) · "
        "[ModelScope model](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled)"
    )


if __name__ == "__main__":
    # ModelScope 容器监听公开 7860 端口；Gradio 队列限制并发调用同一原生进程。
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
