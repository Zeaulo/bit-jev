"""在正式转换前检查蒸馏检查点的每个张量都能映射到 BitNet GGUF 名称。"""

import argparse
import json
import sys
from pathlib import Path


# 从固定的 llama.cpp 子模块加载与转换器一致的 GGUF 映射表。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "learning/bitnet/3rdparty/llama.cpp/gguf-py"))
import gguf


def main():
    """枚举索引中的张量名并报告无法映射的条目。"""
    # 输入路径必须指向需要导出的完整检查点。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    index = json.loads((run / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET_B158, config["num_hidden_layers"])
    missing = []
    # 转换器仅对裸骨干层额外尝试 model. 前缀。
    for name in index["weight_map"]:
        mapped = mapping.get_name(key=name, try_suffixes=(".weight", ".bias"))
        if mapped is None and name.startswith("layers."):
            mapped = mapping.get_name(key="model." + name, try_suffixes=(".weight", ".bias"))
        if mapped is None:
            missing.append(name)
    if missing:
        raise AssertionError(f"{len(missing)} 个张量无法映射：{missing[:12]}")
    print(f"张量映射通过：{len(index['weight_map'])} / {len(index['weight_map'])}")


if __name__ == "__main__":
    main()
