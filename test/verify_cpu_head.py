"""核对原生 CPU 指针头文件与训练检查点完全一致。"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch


# 测试脚本位于项目根目录，显式加入 core 以复用正式检查点读取逻辑。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bit_jev.checkpoint import read_meta  # noqa: E402


def verify(run, artifact):
    """逐字段比较二进制参数，并检查 manifest 中的摘要与大小。"""
    run = Path(run)
    artifact = Path(artifact)
    payload = (artifact / "head.f32").read_bytes()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["head_f32_file"] == "head.f32"
    assert manifest["head_f32_bytes"] == len(payload)
    assert manifest["head_f32_sha256"] == hashlib.sha256(payload).hexdigest()
    # 文件头与蒸馏骨干配置、检查点温度逐一核对。
    magic, hidden_size, head_size, temperature = struct.unpack_from("<8sIIf", payload)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    meta = read_meta(run)
    assert magic == b"BJHEAD01"
    assert hidden_size == config["hidden_size"]
    assert head_size == meta.head_dim
    assert abs(temperature - meta.temperature) < 1e-6
    # 每一段权重必须与 head.pt 的 float32 行优先数组逐字节一致。
    offset = struct.calcsize("<8sIIf")
    for name in ("q.weight", "q.bias", "k.weight", "k.bias"):
        values = meta.head[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        expected = values.numpy().tobytes(order="C")
        actual = payload[offset:offset + len(expected)]
        assert actual == expected, f"参数不一致：{name}"
        offset += len(expected)
    assert offset == len(payload), "存在未定义的尾部字节"
    print(f"head.f32 验证通过：{len(payload)} 字节，4 组参数")


def main():
    """读取本地蒸馏目录与导出目录的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    verify(args.run, args.artifact)


if __name__ == "__main__":
    main()
