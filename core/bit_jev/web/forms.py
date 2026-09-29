"""本地与公开 Space 共用的三题型 Gradio 表单。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gradio as gr


def toggle_description(opened: bool) -> tuple[bool, dict]:
    """展开或收起当前项目下方的整行说明输入。"""
    next_opened = not opened
    return next_opened, gr.update(visible=next_opened)


def description_input(label: str, language: str) -> gr.Textbox:
    """创建宽度随表单变化的说明输入，并由右侧按钮控制显隐。"""
    chinese = language == "zh"
    return gr.Textbox(label=f"{label} 说明（可选）" if chinese else
                      f"{label} description (optional)",
                      placeholder="留空则复制本项内容" if chinese else "Defaults to this item’s text",
                      lines=2, visible=False)


def add_option_rows(kind: str, language: str) -> tuple[list[gr.Textbox], list[gr.Textbox],
                                                         list[gr.Group], list[gr.Button]]:
    """创建二至四个选项／等级，说明输入始终占用完整表单宽度。"""
    chinese = language == "zh"
    defaults = ((["支付与退款", "物流配送", "", ""] if chinese else
                 ["Payments and refunds", "Shipping and delivery", "", ""])
                if kind == "choice" else
                (["低", "中", "高", ""] if chinese else ["Low", "Medium", "High", ""]))
    initial_count = 2 if kind == "choice" else 3
    names: list[gr.Textbox] = []
    descriptions: list[gr.Textbox] = []
    rows: list[gr.Group] = []
    remove_buttons: list[gr.Button] = []
    for index in range(4):
        # 追加按钮逐个显示隐藏的第三、四项；隐藏项不进入推理请求。
        with gr.Group(visible=index < initial_count) as row:
            letter = "ABCD"[index]
            title = "选项" if kind == "choice" else "等级"
            english_title = "Option" if kind == "choice" else "Level"
            with gr.Row(equal_height=True):
                name = gr.Textbox(label=f"{letter}. {title}" if chinese else f"{letter}. {english_title}",
                                  value=defaults[index], lines=1, max_lines=2,
                                  scale=8, min_width=220)
                toggle = gr.Button("＋ 说明" if chinese else "+ Details",
                                   scale=1, min_width=96, variant="secondary")
                remove = gr.Button("删除" if chinese else "Remove", scale=1,
                                   min_width=82, variant="secondary",
                                   visible=initial_count > 2)
            description = description_input(letter, language)
            opened = gr.State(False)
            toggle.click(toggle_description, inputs=[opened], outputs=[opened, description],
                         queue=False)
        names.append(name)
        descriptions.append(description)
        rows.append(row)
        remove_buttons.append(remove)
    return names, descriptions, rows, remove_buttons


def add_noul_rows(language: str) -> list[gr.Textbox]:
    """显示固定的否／是两项，允许独立展开完整宽度的说明。"""
    chinese = language == "zh"
    labels = ["否", "是"] if chinese else ["No", "Yes"]
    descriptions: list[gr.Textbox] = []
    for index, label in enumerate(labels):
        with gr.Group():
            with gr.Row(equal_height=True):
                gr.Textbox(label=f"{'AB'[index]}. {label}", value=label,
                           interactive=False, scale=8, min_width=220)
                toggle = gr.Button("＋ 说明" if chinese else "+ Details",
                                   scale=1, min_width=112, variant="secondary")
            description = description_input(label, language)
            opened = gr.State(False)
            toggle.click(toggle_description, inputs=[opened], outputs=[opened, description],
                         queue=False)
        descriptions.append(description)
    return descriptions


def show_next_option(count: float) -> tuple:
    """依次显示第三、四项，到四项后禁用添加按钮。"""
    next_count = min(int(count) + 1, 4)
    return (next_count, gr.update(visible=next_count >= 3),
            gr.update(visible=next_count >= 4), gr.update(interactive=next_count < 4),
            *(gr.update(visible=next_count > 2) for _index in range(4)))


def remove_option(index: int, count: float, *values: str) -> tuple:
    """删除任意一项，把后续文本依次前移，始终至少保留两项。"""
    active_count = int(count)
    names = list(values[:4])
    descriptions = list(values[4:8])
    if 2 < active_count <= 4 and index < active_count:
        for position in range(index, active_count - 1):
            # 保持选项与说明一同前移；字母 A-D 由表单位置决定。
            names[position] = names[position + 1]
            descriptions[position] = descriptions[position + 1]
        names[active_count - 1] = ""
        descriptions[active_count - 1] = ""
        active_count -= 1
    return (active_count, *names, *descriptions,
            gr.update(visible=active_count >= 3),
            gr.update(visible=active_count >= 4),
            gr.update(interactive=active_count < 4),
            *(gr.update(visible=active_count > 2) for _index in range(4)))


def add_form(kind: str, language: str, infer_callback: Callable) -> gr.Group:
    """按问题、选项、提交、结果的单列顺序创建本地化表单。"""
    chinese = language == "zh"
    with gr.Group(visible=chinese, elem_classes=["decision-form"]) as panel:
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
            names, descriptions, rows, remove_buttons = add_option_rows(kind, language)
            # 隐藏 Number 允许网页按钮和公开 API 提交相同的项目数。
            count = gr.Number(value=2 if kind == "choice" else 3,
                              precision=0, visible=False)
            add = gr.Button("＋ 添加选项（最多 4 个）" if chinese and kind == "choice" else
                            "＋ 添加等级（最多 4 级）" if chinese else
                            "+ Add option (max 4)" if kind == "choice" else
                            "+ Add level (max 4)", variant="secondary")
            add.click(show_next_option, inputs=[count],
                      outputs=[count, rows[2], rows[3], add, *remove_buttons], queue=False)
            for index, remove in enumerate(remove_buttons):
                # 删除任何可见项目后，把后面的文字和说明向前补位。
                remove.click(partial(remove_option, index),
                             inputs=[count, *names, *descriptions],
                             outputs=[count, *names, *descriptions,
                                      rows[2], rows[3], add, *remove_buttons],
                             queue=False)
            inputs = [count, state, question, *names, *descriptions]
        run = gr.Button("运行真实推理" if chinese else "Run real inference", variant="primary")
        # 结果先给简明结论，再给可扫读的概率条；原始 JSON 仅按需查看。
        summary = gr.Markdown(elem_classes=["decision-summary"])
        chart = gr.HTML()
        with gr.Accordion("原始结果（高级）" if chinese else "Raw result (advanced)", open=False):
            detail = gr.JSON(label="完整推理结果" if chinese else "Full inference result")
        run.click(infer_callback, inputs=inputs, outputs=[summary, chart, detail],
                  api_name=f"infer_{kind}_{language}", concurrency_limit=1)
    return panel
