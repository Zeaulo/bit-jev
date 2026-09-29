"""验证三个公开表单的请求默认值、四项上限和结果格式。"""

import sys
import unittest
from pathlib import Path


# 待测纯逻辑位于 Space 源码目录；测试脚本始终留在项目根目录 test/。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spaces" / "bit_jev_demo"))
from logic import build_choice_request, build_noul_request, build_score_request, present_result  # noqa: E402


class SpaceDemoTest(unittest.TestCase):
    """检查 Choice、Noul、Score 的结构化请求契约。"""

    def test_choice_defaults_and_four_options(self):
        """背景和说明缺省时使用规定值，并接受最多四个候选项。"""
        request = build_choice_request("", "哪个团队？", ["支付", "物流"], ["", "送货"])
        self.assertEqual(request["state"], "常见问题")
        self.assertEqual(request["questions"]["decision"]["criteria"],
                         {"支付": "支付", "物流": "送货"})
        four = build_choice_request("背景", "问题", ["A", "B", "C", "D"], ["", "", "", ""])
        self.assertEqual(len(four["questions"]["decision"]["criteria"]), 4)
        with self.assertRaises(ValueError):
            build_choice_request("", "问题", ["A", "B", "C", "D", "E"], [""] * 5)
        with self.assertRaises(ValueError):
            build_choice_request("", "问题", ["A", "A"], ["", ""])

    def test_noul_defaults(self):
        """是非题始终传递 false 和 true，空说明复制语言对应标签。"""
        chinese = build_noul_request("", "是否退款？", "zh", "", "")
        english = build_noul_request("", "Refund?", "en", "", "Absolutely")
        self.assertEqual(chinese["questions"]["decision"]["criteria"],
                         {"false": "否", "true": "是"})
        self.assertEqual(english["questions"]["decision"]["criteria"],
                         {"false": "No", "true": "Absolutely"})

    def test_score_defaults_and_limit(self):
        """等级按显示顺序传递，缺省说明复制等级，超过四级拒绝。"""
        request = build_score_request("", "紧急程度？", ["低", "中", "高"], ["", "", "非常紧急"])
        self.assertEqual(request["questions"]["decision"]["criteria"], [
            {"level": "低", "description": "低"},
            {"level": "中", "description": "中"},
            {"level": "高", "description": "非常紧急"},
        ])
        four = build_score_request("", "Q", ["1", "2", "3", "4"], [""] * 4)
        self.assertEqual(len(four["questions"]["decision"]["criteria"]), 4)
        with self.assertRaises(ValueError):
            build_score_request("", "Q", ["1", "2", "3", "4", "5"], [""] * 5)

    def test_result_types(self):
        """三种答案分别显示选项、是的概率和期望等级。"""
        answers = [
            ({"type": "choice", "choice": "支付", "probabilities": {"支付": .8, "物流": .2}}, "支付"),
            ({"type": "noul", "noul": .75}, "75.0%"),
            ({"type": "score", "score": 1.25, "probabilities": {"0": .1, "1": .6, "2": .3},
              "legend": {"0": "低", "1": "中", "2": "高"}}, "1.25"),
        ]
        for answer, expected in answers:
            with self.subTest(kind=answer["type"]):
                summary, detail = present_result(
                    {"answers": {"decision": answer}, "latency_ms": 123.45, "device": "cpu"}, "zh")
                self.assertIn(expected, summary)
                self.assertEqual(detail["latency_ms"], 123.45)


if __name__ == "__main__":
    unittest.main()
