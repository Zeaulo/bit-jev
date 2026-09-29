"""Hugging Face Space 与 ModelScope 创空间共用的双语 CPU 推理页面。"""

from __future__ import annotations

import os
import threading

import gradio as gr
from bit_jev.gguf import BitJev

from logic import build_request, present_result


# 模型进程常驻并串行接收 JSONL；公共空间限制为一次执行一条请求。
_model: BitJev | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_model() -> BitJev:
    """首次请求才下载模型，之后复用同一个原生进程。"""
    global _model
    with _model_lock:
        if _model is None:
            # Docker 镜像在构建阶段完成 Linux 原生编译，运行时仅需加载权重。
            _model = BitJev.from_pretrained(
                device="cpu",
                threads=min(os.cpu_count() or 2, 2),
                source=os.environ.get("BIT_JEV_MODEL_SOURCE", "auto"),
                binary=os.environ.get("BIT_JEV_BINARY", "/opt/bit-jev-cpu"),
            )
        return _model


def infer(language: str, state: str, question: str, first_key: str,
          first_description: str, second_key: str,
          second_description: str) -> tuple[str, dict]:
    """对页面输入执行真实推理，并把可恢复错误显示给访问者。"""
    try:
        # 在载入模型前校验输入，避免无效请求触发约 1.19 GB 下载。
        request = build_request(state, question, first_key, first_description,
                                second_key, second_description)
        with _infer_lock:
            result = get_model().infer(request)
        return present_result(result, language)
    except Exception as error:
        label = "运行失败" if language == "zh" else "Inference failed"
        return f"{label}: `{type(error).__name__}: {error}`", {}


def add_language_tab(language: str) -> None:
    """建立一套本地化表单；两页使用相同模型与输入契约。"""
    chinese = language == "zh"
    with gr.Tab("中文" if chinese else "English"):
        gr.Markdown(
            "### 普通电脑，CPU 就能跑 JEV，而且很快！\n修改下面的客服案例，然后运行真实模型。"
            if chinese else
            "### Run structured JEV decisions on a CPU\nEdit the customer support example and run the real model."
        )
        with gr.Row():
            with gr.Column(scale=3):
                state = gr.Textbox(
                    label="情境 / state" if chinese else "Context / state",
                    value="客户报告同一订单被重复扣款。" if chinese else
                          "A customer reports being charged twice for the same order.",
                    lines=3,
                )
                question = gr.Textbox(
                    label="判断问题" if chinese else "Decision question",
                    value="哪个团队应该处理？" if chinese else "Which team should handle this?",
                )
                with gr.Row():
                    first_key = gr.Textbox(label="选项 A 标识" if chinese else "Option A key",
                                           value="billing")
                    first_description = gr.Textbox(
                        label="选项 A 说明" if chinese else "Option A description",
                        value="支付与退款" if chinese else "Payments and refunds",
                    )
                with gr.Row():
                    second_key = gr.Textbox(label="选项 B 标识" if chinese else "Option B key",
                                            value="shipping")
                    second_description = gr.Textbox(
                        label="选项 B 说明" if chinese else "Option B description",
                        value="物流配送" if chinese else "Shipping and delivery",
                    )
                run = gr.Button("运行真实推理" if chinese else "Run real inference", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown("等待提交。" if chinese else "Ready to run.")
                detail = gr.JSON(label="答案与概率" if chinese else "Answer and probabilities")
        # Gradio 队列防止同一常驻 C++ 进程被并发写入。
        run.click(lambda *values: infer(language, *values),
                  inputs=[state, question, first_key, first_description,
                          second_key, second_description],
                  outputs=[summary, detail], concurrency_limit=1)


with gr.Blocks(title="bit-jev · Live CPU Demo", theme=gr.themes.Soft()) as demo:
    gr.Image(value="logo.png", show_label=False, height=160, interactive=False, container=False)
    gr.Markdown("# bit-jev 🚀 0 · 1\nBitNet · LoRA · teacher–student distillation · I2_S GGUF")
    add_language_tab("zh")
    add_language_tab("en")
    gr.Markdown(
        "**说明 / Note:** 首次运行需下载约 1.19 GB 模型并加载到内存，耗时取决于网络。"
        "页面报告的 `latency_ms` 仅为原生计算时间；免费空间为 2 vCPU，"
        "不能拿它与 README 的 32 核 EPYC 或 RTX 5090 测量直接比较。"
        " / The first run downloads and loads ~1.19 GB. The reported latency excludes that time. "
        "This free 2-vCPU Space is not comparable with the published 32-core EPYC or RTX 5090 tests.\n\n"
        "[GitHub](https://github.com/Zeaulo/bit-jev) · "
        "[Hugging Face model](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) · "
        "[ModelScope model](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled)"
    )


if __name__ == "__main__":
    # 两个平台的容器都监听公开 7860 端口，不向页面暴露令牌或私有路径。
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
