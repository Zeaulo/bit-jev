"""验证安装即用入口不需要输入文件且输出真实推理结果。"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# 测试优先导入当前项目，而不是机器中可能安装的旧版 wheel。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from bit_jev.demo import DEMO_REQUEST, main  # noqa: E402


class DemoTests(unittest.TestCase):
    """检查命令入口的模型参数与 JSON 输出契约。"""

    def test_runs_without_jsonl_and_prints_inference(self):
        """用户只给设备参数即可运行一条自带请求。"""
        # 模拟模型只用于核对命令编排，实际 GGUF 推理由发行验收另行覆盖。
        model = MagicMock()
        model.__enter__.return_value = model
        model.infer.return_value = {"answers": {"team": "billing"},
                                    "latency_ms": 12.5, "device": "vulkan"}
        output = io.StringIO()
        with patch("bit_jev.demo.BitJev.from_pretrained", return_value=model) as load_model, \
             contextlib.redirect_stdout(output):
            main(["--model", "example/local", "--device", "gpu", "--threads", "8"])
        load_model.assert_called_once_with("example/local", device="gpu", threads=8, gpu_index=None)
        model.infer.assert_called_once_with(DEMO_REQUEST)
        self.assertEqual(json.loads(output.getvalue())["answers"], {"team": "billing"})


if __name__ == "__main__":
    unittest.main()
