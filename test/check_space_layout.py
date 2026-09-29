"""离线检查 Gradio 页面恰好暴露中英文三题型推理入口。"""

import sys
from pathlib import Path


# 优先加载工作区中的包源码，再加载 Space 薄入口；不触发模型下载。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spaces" / "bit_jev_demo"))
from app import demo  # noqa: E402


def main() -> None:
    """核对六条固定 API、输入数量、三组顶层任务 Tab。"""
    expected = {f"infer_{kind}_{language}": (4 if kind == "noul" else 11)
                for kind in ("choice", "noul", "score")
                for language in ("zh", "en")}
    dependencies = demo.config["dependencies"]
    actual = {item.get("api_name"): len(item.get("inputs", []))
              for item in dependencies if item.get("api_name") in expected}
    assert actual == expected, (actual, expected)
    for item in dependencies:
        if item.get("api_name") in expected:
            assert len(item.get("outputs", [])) == 3, item["api_name"]
    # 四项计数必须由可提交的隐藏 Number 承载；State 会忽略公开 API 的输入值。
    component_types = {item["id"]: item["type"] for item in demo.config["components"]}
    for item in dependencies:
        if item.get("api_name", "").startswith(("infer_choice_", "infer_score_")):
            assert component_types[item["inputs"][0]] == "number", item["api_name"]
    tabs = [item["props"].get("label") for item in demo.config["components"]
            if item.get("type") == "tabitem"]
    assert tabs == ["Choice · 选择题", "Noul · 是非题", "Score · 等级题"], tabs
    print("三个任务 Tab、六种语言入口与输入数量均正确")


if __name__ == "__main__":
    main()
