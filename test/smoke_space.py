"""向公开 ModelScope Gradio 空间提交手写案例，核对真实推理结果。"""

from __future__ import annotations

import json
import sys

import requests


# ModelScope 的浏览器访问域名按浏览器 User-Agent 分流；该脚本不使用发布令牌。
BASE_URL = "https://jinghao9616-bit-jev-demo.ms.show"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
}
CASE = ["客户报告同一订单被重复扣款。", "哪个团队应该处理？",
        "billing", "支付与退款", "shipping", "物流配送"]
ENGLISH_CASE = ["A customer reports being charged twice for the same order.",
                "Which team should handle this?", "billing", "Payments and refunds",
                "shipping", "Shipping and delivery"]


def check_case(api_name: str, case: list[str]) -> None:
    """调用指定语言的 Gradio 事件，并确认返回 choice 与正耗时。"""
    # 两个标签页各有回调；其名称由 /config 的公共元数据确认。
    response = requests.get(f"{BASE_URL}/config", headers=HEADERS, timeout=30)
    response.raise_for_status()
    dependencies = response.json()["dependencies"]
    if not any(item.get("api_name") == api_name and len(item.get("inputs", [])) == 6
               for item in dependencies):
        raise RuntimeError(f"线上页面没有预期的六字段 {api_name} 推理回调")
    # Gradio call 接口返回事件 ID；实际答案从该事件的 SSE 流读取。
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
            if not isinstance(detail, dict) or detail.get("answer") not in {"billing", "shipping"}:
                raise RuntimeError(f"线上推理未返回真实候选答案：{summary}")
            if float(detail.get("latency_ms", 0)) <= 0:
                raise RuntimeError("线上推理未返回正数原生耗时")
            print(json.dumps({"language": api_name, "summary": summary,
                              "detail": detail}, ensure_ascii=False))
            return
    raise RuntimeError("线上空间未在时限内返回推理结果")


def main() -> None:
    """分别验证中英文标签页确实调用同一个真实模型。"""
    check_case("lambda", CASE)
    check_case("lambda_1", ENGLISH_CASE)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"线上验收失败：{type(error).__name__}: {error}", file=sys.stderr)
        raise
