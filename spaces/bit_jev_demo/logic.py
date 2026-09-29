"""在线演示的输入校验和结果整理；不依赖 Gradio。"""

from __future__ import annotations

from typing import Any


def build_request(state: str, question: str, first_key: str, first_description: str,
                  second_key: str, second_description: str) -> dict[str, Any]:
    """把双语表单转换成现有 BitJev.infer 接受的 choice 请求。"""
    # 六个字段都由访问者输入；先限制长度，避免公共 CPU 空间被超长请求占满。
    fields = {
        "state": (state, 1200),
        "question": (question, 300),
        "first_key": (first_key, 40),
        "first_description": (first_description, 200),
        "second_key": (second_key, 40),
        "second_description": (second_description, 200),
    }
    cleaned: dict[str, str] = {}
    for name, (value, maximum) in fields.items():
        # 候选标识和描述的空白同样会影响结构化答案，统一剔除首尾空白。
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 不能为空 / cannot be empty")
        cleaned[name] = value.strip()
        if len(cleaned[name]) > maximum:
            raise ValueError(f"{name} 最多 {maximum} 个字符 / max {maximum} characters")
    if cleaned["first_key"] == cleaned["second_key"]:
        raise ValueError("两个候选标识必须不同 / option keys must differ")
    return {
        "state": cleaned["state"],
        "questions": {
            "team": {
                "type": "choice",
                "instructions": cleaned["question"],
                "criteria": {
                    cleaned["first_key"]: cleaned["first_description"],
                    cleaned["second_key"]: cleaned["second_description"],
                },
            },
        },
    }


def present_result(result: dict[str, Any], language: str) -> tuple[str, dict[str, Any]]:
    """展示模型实际返回的答案、概率和原生计算耗时。"""
    # 推理结果由 bit_jev.gguf.BitJev.infer 生成；不注入预设答案或模拟耗时。
    answer = result["answers"]["team"]
    choice = answer["choice"]
    latency = float(result["latency_ms"])
    title = (f"模型选择：**{choice}**" if language == "zh"
             else f"Model choice: **{choice}**")
    timing = (f"CPU 原生计算：**{latency:.2f} ms**（不含首次下载与加载）"
              if language == "zh" else
              f"Native CPU compute: **{latency:.2f} ms** (excludes download and loading)")
    return f"{title}\n\n{timing}", {
        "answer": choice,
        "probabilities": answer["probabilities"],
        "latency_ms": latency,
        "device": result["device"],
    }
