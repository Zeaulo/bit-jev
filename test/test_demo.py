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

    def test_once_prints_inference(self):
        """显式 --once 继续运行一条自带请求并输出 JSON。"""
        # 模拟模型只用于核对命令编排，实际 GGUF 推理由发行验收另行覆盖。
        model = MagicMock()
        model.__enter__.return_value = model
        model.infer.return_value = {"answers": {"team": "billing"},
                                    "latency_ms": 12.5, "device": "vulkan"}
        output = io.StringIO()
        with patch("bit_jev.demo.BitJev.from_pretrained", return_value=model) as load_model, \
             contextlib.redirect_stdout(output):
            main(["--once", "--model", "example/local", "--device", "gpu", "--threads", "8"])
        load_model.assert_called_once_with("example/local", source="auto", device="gpu",
                                           threads=8, gpu_index=None, binary=None)
        model.infer.assert_called_once_with(DEMO_REQUEST)
        self.assertEqual(json.loads(output.getvalue())["answers"], {"team": "billing"})

    def test_default_starts_local_web_without_loading_model(self):
        """默认入口只启动回环地址网页，加载模型发生在第一次有效提交。"""
        with patch("bit_jev.web.app.launch_local") as launch, \
             patch("bit_jev.demo.BitJev.from_pretrained") as load_model:
            main(["--source", "modelscope", "--threads", "8", "--no-browser"])
        self.assertEqual(launch.call_args.kwargs,
                         {"host": "127.0.0.1", "port": 7860, "open_browser": False})
        self.assertEqual(launch.call_args.args[0].source, "modelscope")
        self.assertEqual(launch.call_args.args[0].threads, 8)
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
