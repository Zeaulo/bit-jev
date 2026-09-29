"""检查三题型概率条的内容与 HTML 文本转义。"""

import sys
import unittest
from pathlib import Path


# 始终验证当前项目源码，不读取旧版已安装 wheel。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from bit_jev.web.presentation import render_probabilities  # noqa: E402


class WebPresentationTests(unittest.TestCase):
    """确保结果可读，并避免用户输入进入 HTML 结构。"""

    def test_choice_escapes_user_option(self):
        """Choice 选项名包含标签时，HTML 只显示转义后的文本。"""
        chart = render_probabilities({"type": "choice", "answer": "<script>",
                                      "probabilities": {"<script>": 0.75, "B": 0.25}}, "zh")
        self.assertIn("&lt;script&gt;", chart)
        self.assertNotIn("<script>", chart)
        self.assertIn("75.0%", chart)

    def test_noul_and_score_have_human_labels(self):
        """是非题显示否和是；等级题不把内部 level 字段名暴露为标题。"""
        noul = render_probabilities({"type": "noul", "answer": 0.6}, "zh")
        score = render_probabilities({"type": "score", "answer": 1.2,
                                      "probabilities": {"0": .3, "1": .7},
                                      "legend": {"0": "level: 低\ndescription: 低",
                                                 "1": "level: 高\ndescription: 高"}}, "zh")
        self.assertIn("否", noul)
        self.assertIn("是", noul)
        self.assertIn("40.0%", noul)
        self.assertIn("高", score)
        self.assertNotIn("level: 高", score)


if __name__ == "__main__":
    unittest.main()
