"""检查蒸馏检查点分片与 GGUF 转换器实际读取的张量数量。"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main():
    """输出索引张量数、各分片张量数及少量名称样本。"""
    # 输入路径由命令行指定，避免隐式读取别的模型目录。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    index = json.loads((run / "model.safetensors.index.json").read_text(encoding="utf-8"))
    print("索引张量数：", len(index["weight_map"]))
    # 分片文件名由索引提供，并逐个实际打开。
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(run / filename, framework="pt", device="cpu") as source:
            names = list(source.keys())
        print(filename, "张量数：", len(names), "名称样本：", names[:3])


if __name__ == "__main__":
    main()
