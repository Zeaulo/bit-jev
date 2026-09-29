"""在线演示的三种请求转换与结果整理；不依赖 Gradio。"""

from __future__ import annotations

from typing import Any


def clean_text(value: str | None, name: str, maximum: int, required: bool = True) -> str:
    """可选的空组件值按空文本处理，必填字段继续校验类型与长度。"""
    # Gradio 5 的折叠 Textbox 初始值可能是 None，语义上等同于未填写。
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是文本 / must be text")
    cleaned = value.strip()
    if required and not cleaned:
        raise ValueError(f"{name} 不能为空 / cannot be empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{name} 最多 {maximum} 个字符 / max {maximum} characters")
    return cleaned


def base_request(state: str | None, question: str, kind: str, criteria: Any) -> dict[str, Any]:
    """构造三种题型共用的单题 SystemOne 请求。"""
    # 背景未填写时向模型明确传入“常见问题”，保持界面与 API 的默认值一致。
    background = clean_text(state, "背景说明", 1200, required=False) or "常见问题"
    instruction = clean_text(question, "你的问题", 300)
    return {"state": background, "questions": {"decision": {
        "type": kind, "instructions": instruction, "criteria": criteria,
    }}}


def build_choice_request(state: str | None, question: str,
                         options: list[str], descriptions: list[str | None]) -> dict[str, Any]:
    """把二至四个选项转换为 choice；空说明复制选项内容。"""
    if len(options) != len(descriptions) or not 2 <= len(options) <= 4:
        raise ValueError("选择题需要 2 至 4 个选项 / choice needs 2 to 4 options")
    # 字典键是用户实际看见的选项，答案可直接在页面上解释。
    criteria: dict[str, str] = {}
    for index, (option, description) in enumerate(zip(options, descriptions)):
        label = clean_text(option, f"选项 {index + 1}", 80)
        detail = clean_text(description, f"选项 {index + 1} 说明", 200, required=False)
        if label in criteria:
            raise ValueError("选项内容不能重复 / option text must be unique")
        criteria[label] = detail or label
    return base_request(state, question, "choice", criteria)


def build_noul_request(state: str | None, question: str, language: str,
                       no_description: str | None, yes_description: str | None) -> dict[str, Any]:
    """把固定的否／是选项转换为 noul；空说明复制显示标签。"""
    labels = ("否", "是") if language == "zh" else ("No", "Yes")
    no_detail = clean_text(no_description, "否说明", 200, required=False) or labels[0]
    yes_detail = clean_text(yes_description, "是说明", 200, required=False) or labels[1]
    return base_request(state, question, "noul", {"false": no_detail, "true": yes_detail})


def build_score_request(state: str | None, question: str,
                        levels: list[str], descriptions: list[str | None]) -> dict[str, Any]:
    """把二至四个有序等级转换为 score；空说明复制等级文本。"""
    if len(levels) != len(descriptions) or not 2 <= len(levels) <= 4:
        raise ValueError("等级题需要 2 至 4 级 / score needs 2 to 4 levels")
    criteria: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, (level, description) in enumerate(zip(levels, descriptions)):
        label = clean_text(level, f"等级 {index + 1}", 80)
        detail = clean_text(description, f"等级 {index + 1} 说明", 200, required=False)
        if label in seen:
            raise ValueError("等级内容不能重复 / level text must be unique")
        seen.add(label)
        criteria.append({"level": label, "description": detail or label})
    return base_request(state, question, "score", criteria)


def present_result(result: dict[str, Any], language: str) -> tuple[str, dict[str, Any]]:
    """按返回题型展示真实答案、概率与原生计算时间。"""
    answer = result["answers"]["decision"]
    kind = answer["type"]
    latency = float(result["latency_ms"])
    if kind == "choice":
        value = answer["choice"]
        title = f"模型选择：**{value}**" if language == "zh" else f"Model choice: **{value}**"
    elif kind == "noul":
        value = answer["noul"]
        title = (f"回答“是”的概率：**{value:.1%}**" if language == "zh"
                 else f"Probability of Yes: **{value:.1%}**")
    else:
        value = answer["score"]
        title = (f"期望等级索引：**{value:.2f}**（首级为 0）" if language == "zh"
                 else f"Expected level index: **{value:.2f}** (first level is 0)")
    # 原生路径会返回实际设备，不能把 Vulkan/CUDA 用户的耗时标为 CPU。
    device = str(result["device"]).lower()
    hardware = "GPU" if device in {"gpu", "vulkan", "cuda"} else "CPU"
    timing = (f"{hardware} 原生计算：**{latency:.2f} ms**" if language == "zh"
              else f"Native {hardware} compute: **{latency:.2f} ms**")
    return f"{title}\n\n{timing}", {
        "type": kind, "answer": value, "probabilities": answer.get("probabilities"),
        "legend": answer.get("legend"), "confidence": answer.get("confidence"),
        "latency_ms": latency, "device": result["device"],
    }
