"""把结构化概率转换为安全、易读的 HTML 横条图。"""

from __future__ import annotations

from html import escape
from typing import Any


def display_label(kind: str, key: str, legend: dict[str, str], language: str) -> str:
    """为概率条选取用户能看懂的候选项名称。"""
    if kind == "noul":
        labels = {"false": "否", "true": "是"} if language == "zh" else {"false": "No", "true": "Yes"}
        return labels[key]
    if kind == "score":
        # Score 图例由 api.render 生成，第一行 level: 后面是用户输入的等级名。
        first_line = str(legend.get(key, key)).splitlines()[0]
        return first_line.removeprefix("level: ")
    return key


def probability_rows(detail: dict[str, Any]) -> list[tuple[str, float]]:
    """把三题型的返回值统一为标签与概率序列。"""
    kind = detail.get("type")
    if kind == "noul":
        yes = float(detail["answer"])
        return [("false", 1.0 - yes), ("true", yes)]
    probabilities = detail.get("probabilities") or {}
    return [(str(key), float(value)) for key, value in probabilities.items()]


def render_probabilities(detail: dict[str, Any], language: str) -> str:
    """生成屏幕阅读器可读、转义用户文本的概率条。"""
    if not detail or detail.get("type") not in {"choice", "noul", "score"}:
        return ""
    title = "各项概率" if language == "zh" else "Option probabilities"
    kind = str(detail["type"])
    legend = detail.get("legend") or {}
    bars: list[str] = []
    for key, probability in probability_rows(detail):
        label = escape(display_label(kind, key, legend, language))
        bounded = max(0.0, min(1.0, probability))
        percentage = f"{bounded * 100:.1f}%"
        bars.append(
            '<div class="jev-prob-row">'
            f'<div class="jev-prob-heading"><span>{label}</span><strong>{percentage}</strong></div>'
            f'<div class="jev-prob-track" role="img" aria-label="{label}: {percentage}">'
            f'<div class="jev-prob-fill" style="width:{bounded * 100:.1f}%"></div></div></div>'
        )
    return f'<section class="jev-results" aria-label="{escape(title)}"><h3>{escape(title)}</h3>{"".join(bars)}</section>'
