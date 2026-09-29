"""验证 Choice 与 Score 任意项目删除后的顺序和边界。"""

import sys
import unittest
from pathlib import Path


# 表单回调来自可安装包，测试优先导入本地版本。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from bit_jev.web.forms import remove_option, show_next_option  # noqa: E402


class WebFormTests(unittest.TestCase):
    """添加与删除操作不应让项目数越过二至四的范围。"""

    def test_remove_middle_shifts_text_and_description(self):
        """删除中间项后，后续内容及说明一起移动。"""
        result = remove_option(1, 4, "A", "B", "C", "D", "a", "b", "c", "d")
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1:5], ("A", "C", "D", ""))
        self.assertEqual(result[5:9], ("a", "c", "d", ""))

    def test_two_is_minimum_and_four_is_maximum(self):
        """只剩两项时删除不生效，添加到四项后按钮不可继续使用。"""
        remaining = remove_option(0, 2, "A", "B", "", "", "a", "b", "", "")
        self.assertEqual(remaining[0], 2)
        self.assertEqual(remaining[1:3], ("A", "B"))
        self.assertEqual(show_next_option(4)[0], 4)
        self.assertEqual(show_next_option(3)[0], 4)


if __name__ == "__main__":
    unittest.main()
