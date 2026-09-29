"""只验证页面请求契约；不下载模型或启动网络服务。"""

import sys
import unittest
from pathlib import Path


# 待测逻辑位于独立的 Space 源码目录，测试脚本仍保留在项目根目录 test/。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spaces" / "bit_jev_demo"))
from logic import build_request, present_result  # noqa: E402


class SpaceDemoTest(unittest.TestCase):
    """确保公开表单与现有 choice 推理契约一致。"""

    def test_request_and_result(self):
        """检查输入字段和真实结果展示，不预置推理答案。"""
        request = build_request("重复扣款", "哪个团队？", "billing", "支付", "shipping", "物流")
        self.assertEqual(request["questions"]["team"]["criteria"]["billing"], "支付")
        result = {
            "answers": {"team": {"choice": "billing", "probabilities": {"billing": 0.8, "shipping": 0.2}}},
            "latency_ms": 123.45,
            "device": "cpu",
        }
        summary, detail = present_result(result, "zh")
        self.assertIn("billing", summary)
        self.assertEqual(detail["latency_ms"], 123.45)

    def test_invalid_input(self):
        """拒绝空字段和重复选项标识。"""
        with self.assertRaises(ValueError):
            build_request("", "Q", "a", "A", "b", "B")
        with self.assertRaises(ValueError):
            build_request("S", "Q", "same", "A", "same", "B")


if __name__ == "__main__":
    unittest.main()
