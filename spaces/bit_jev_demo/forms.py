"""Gradio 三题型表单：全局切换语言，按需展开背景和选项说明。"""

from __future__ import annotations

from collections.abc import Callable

import gradio as gr


def add_option_rows(kind: str, language: str) -> tuple[list[gr.Textbox], list[gr.Textbox], list[gr.Group]]:
    """创建最多四行选项／等级，返回输入组件及可控制显隐的容器。"""
    chinese = language == "zh"
    defaults = ((["支付与退款", "物流配送", "", ""] if chinese else
                 ["Payments and refunds", "Shipping and delivery", "", ""])
                if kind == "choice" else
                (["低", "中", "高", ""] if chinese else ["Low", "Medium", "High", ""]))
    initial_count = 2 if kind == "choice" else 3
    names: list[gr.Textbox] = []
    descriptions: list[gr.Textbox] = []
    rows: list[gr.Group] = []
    for index in range(4):
        # 未添加的行隐藏且不参与请求；使用者通过同一个按钮依次展开第三、四行。
        with gr.Group(visible=index < initial_count) as row:
            with gr.Row():
                with gr.Column(scale=8, min_width=220):
                    letter = "ABCD"[index]
                    title = "选项" if kind == "choice" else "等级"
                    english_title = "Option" if kind == "choice" else "Level"
                    name = gr.Textbox(label=f"{letter}. {title}" if chinese else f"{letter}. {english_title}",
                                      value=defaults[index], max_lines=2)
                with gr.Column(scale=1, min_width=92):
                    with gr.Accordion("+", open=False):
                        description = gr.Textbox(
                            label=f"{letter} 说明（可选）" if chinese else f"{letter} description (optional)",
                            placeholder="留空则复制本项内容" if chinese else "Defaults to this item’s text",
                            lines=2,
                        )
        names.append(name)
        descriptions.append(description)
        rows.append(row)
    return names, descriptions, rows


def add_noul_rows(language: str) -> list[gr.Textbox]:
    """为固定否／是选项创建两个可展开的说明输入。"""
    chinese = language == "zh"
    labels = ["否", "是"] if chinese else ["No", "Yes"]
    descriptions: list[gr.Textbox] = []
    for index, label in enumerate(labels):
        with gr.Row():
            with gr.Column(scale=8, min_width=220):
                gr.Markdown(f"**{'AB'[index]}. {label}**")
            with gr.Column(scale=1, min_width=92):
                with gr.Accordion("+", open=False):
                    description = gr.Textbox(
                        label=f"{label}说明（可选）" if chinese else f"{label} description (optional)",
                        placeholder="留空则复制选项内容" if chinese else "Defaults to the option text",
                        lines=2,
                    )
        descriptions.append(description)
    return descriptions


def show_next_option(count: float) -> tuple[int, dict, dict, dict]:
    """只显示下一行，并在第四行出现后禁用添加按钮。"""
    next_count = min(int(count) + 1, 4)
    return (next_count, gr.update(visible=next_count >= 3),
            gr.update(visible=next_count >= 4), gr.update(interactive=next_count < 4))


def add_form(kind: str, language: str, infer_callback: Callable) -> gr.Group:
    """建立一个本地化表单，并绑定一次真实模型推理事件。"""
    chinese = language == "zh"
    with gr.Group(visible=chinese) as panel:
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="你的问题" if chinese else "Your question",
                    value=("哪个团队应该处理？" if chinese else "Which team should handle this?")
                    if kind == "choice" else "",
                    placeholder="请输入需要判断的问题" if chinese else "Enter your decision question",
                )
                with gr.Accordion("补充背景（可选）" if chinese else "Add background (optional)", open=False):
                    state = gr.Textbox(label="背景说明" if chinese else "Background", lines=3)
                if kind == "noul":
                    descriptions = add_noul_rows(language)
                    inputs = [state, question, *descriptions]
                else:
                    names, descriptions, rows = add_option_rows(kind, language)
                    # 隐藏数值组件在网页操作和公开 API 中保持相同的项目数契约。
                    count = gr.Number(value=2 if kind == "choice" else 3,
                                      precision=0, visible=False)
                    add = gr.Button("+ 添加选项（最多 4 个）" if chinese and kind == "choice" else
                                    "+ 添加等级（最多 4 级）" if chinese else
                                    "+ Add option (max 4)" if kind == "choice" else
                                    "+ Add level (max 4)", variant="secondary")
                    add.click(show_next_option, inputs=[count],
                              outputs=[count, rows[2], rows[3], add], queue=False)
                    inputs = [count, state, question, *names, *descriptions]
                run = gr.Button("运行真实推理" if chinese else "Run real inference", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown("等待提交。" if chinese else "Ready to run.")
                detail = gr.JSON(label="答案与概率" if chinese else "Answer and probabilities")
        # 固定 API 名称，方便两个公开空间及验收脚本准确定位六种表单。
        run.click(infer_callback, inputs=inputs, outputs=[summary, detail],
                  api_name=f"infer_{kind}_{language}", concurrency_limit=1)
    return panel
