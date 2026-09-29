"""通过公开 Gradio API 验收中英文三题型的真实在线推理。"""

from __future__ import annotations

import argparse
import json
import sys

import requests


# 运行域名与浏览器 User-Agent 对齐；此脚本不使用发布令牌。
BASE_URL = "https://jinghao9616-bit-jev-demo.ms.show"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
}


def cases() -> list[tuple[str, list[object], str]]:
    """返回六个涵盖两种语言与三种题型的真实请求。"""
    return [
        ("infer_choice_zh", [2, "", "哪个团队处理重复扣款？", "支付", "物流", "", "", "", "", "", ""], "choice"),
        ("infer_choice_en", [2, "", "Which team handles a duplicate charge?", "Billing", "Shipping", "", "", "", "", "", ""], "choice"),
        ("infer_noul_zh", ["", "是否应该退款？", "", ""], "noul"),
        ("infer_noul_en", ["", "Should we issue a refund?", "", ""], "noul"),
        ("infer_score_zh", [3, "", "这件事有多紧急？", "低", "中", "高", "", "", "", "", ""], "score"),
        ("infer_score_en", [3, "", "How urgent is this?", "Low", "Medium", "High", "", "", "", "", ""], "score"),
    ]


def four_item_cases() -> list[tuple[str, list[object], str]]:
    """用四个有效项目验证 Choice 和 Score 的上限路径。"""
    return [
        ("infer_choice_zh", [4, "", "哪个团队处理？", "支付", "物流", "客服", "风控",
                             "", "", "", ""], "choice"),
        ("infer_score_zh", [4, "", "紧急程度？", "低", "中", "高", "极高",
                            "", "", "", ""], "score"),
    ]


def check_case(api_name: str, case: list[object], kind: str,
               dependencies: list[dict]) -> None:
    """提交单条在线请求，核对答案类型和正数原生耗时。"""
    if not any(item.get("api_name") == api_name and len(item.get("inputs", [])) == len(case)
               for item in dependencies):
        raise RuntimeError(f"线上页面缺少 {api_name} 或输入数量不符")
    response = requests.post(f"{BASE_URL}/gradio_api/call/{api_name}",
                             headers=HEADERS, json={"data": case}, timeout=40)
    response.raise_for_status()
    event_id = response.json()["event_id"]
    with requests.get(f"{BASE_URL}/gradio_api/call/{api_name}/{event_id}", headers=HEADERS,
                      stream=True, timeout=(30, 900)) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            if not isinstance(payload, list) or len(payload) != 2:
                continue
            summary, detail = payload
            if not isinstance(detail, dict) or detail.get("type") != kind:
                raise RuntimeError(f"{api_name} 未返回预期题型：{summary}")
            if kind in ("choice", "score") and len(detail.get("probabilities") or {}) != int(case[0]):
                raise RuntimeError(f"{api_name} 未计算请求中的全部 {case[0]} 个项目")
            if float(detail.get("latency_ms", 0)) <= 0:
                raise RuntimeError(f"{api_name} 未返回正数原生耗时")
            print(json.dumps({"api": api_name, "summary": summary,
                              "detail": detail}, ensure_ascii=False))
            return
    raise RuntimeError(f"{api_name} 未在时限内返回结果")


def main() -> None:
    """确认公开端点存在，然后逐个验收六种界面路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-only", action="store_true", help="仅测试两种四项上限路径")
    args = parser.parse_args()
    response = requests.get(f"{BASE_URL}/config", headers=HEADERS, timeout=30)
    response.raise_for_status()
    dependencies = response.json()["dependencies"]
    selected = four_item_cases() if args.four_only else cases()
    for api_name, case, kind in selected:
        check_case(api_name, case, kind, dependencies)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"线上验收失败：{type(error).__name__}: {error}", file=sys.stderr)
        raise
