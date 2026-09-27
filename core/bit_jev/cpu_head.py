"""将 PyTorch pointer head 转换为供原生 CPU runner 读取的定长 float32 文件。"""

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import torch

from .checkpoint import BitJevCheckpoint


# 文件头依次包含标识、骨干维度、指针维度和推理温度。
MAGIC = b"BJHEAD01"
HEADER = struct.Struct("<8sIIf")
ORDER = ("q.weight", "q.bias", "k.weight", "k.bias")


def encode_head(meta, hidden_size):
    """验证指针头形状，并按固定次序序列化所有参数。"""
    # 指针维度来自训练检查点，骨干维度来自模型配置。
    head_dim = int(meta.head_dim)
    hidden_size = int(hidden_size)
    temperature = float(meta.temperature)
    if head_dim <= 0 or hidden_size <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("指针头维度或温度无效")
    # 四个矩阵和偏置的形状严格固定，防止写出错位的二进制文件。
    expected_shapes = {
        "q.weight": (head_dim, hidden_size),
        "q.bias": (head_dim,),
        "k.weight": (head_dim, hidden_size),
        "k.bias": (head_dim,),
    }
    payload = bytearray(HEADER.pack(MAGIC, hidden_size, head_dim, temperature))
    for name in ORDER:
        tensor = meta.head.get(name)
        if tensor is None or tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(f"指针头参数缺失或形状不符：{name}")
        # PyTorch 张量按 CPU、float32、C 连续顺序写出。
        values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if not bool(torch.isfinite(torch.from_numpy(values)).all()):
            raise ValueError(f"指针头参数包含非有限值：{name}")
        payload.extend(values.tobytes(order="C"))
    return bytes(payload)


def write_head_binary(run, artifact):
    """写入指针头 sidecar，并把文件摘要加入既有清单。"""
    # 检查点负责读取与训练时相同的头参数。
    checkpoint = BitJevCheckpoint(run)
    config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    blob = encode_head(checkpoint.meta, config["hidden_size"])
    artifact = Path(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(f"导出目录不存在：{artifact}")
    # 先写临时文件，再原子替换，避免意外中断留下半个指针头。
    destination = artifact / "head.f32"
    temporary = artifact / "head.f32.partial"
    temporary.write_bytes(blob)
    os.replace(temporary, destination)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["head_f32_file"] = destination.name
    manifest["head_f32_bytes"] = len(blob)
    manifest["head_f32_sha256"] = hashlib.sha256(blob).hexdigest()
    # 清单与 sidecar 同步更新，供 CPU runner 和测试读取。
    temporary_manifest = artifact / "manifest.json.partial"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return destination


def main(argv=None):
    """从命令行给现有蒸馏导出目录添加 CPU 指针头。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    print(write_head_binary(args.run, args.artifact))


if __name__ == "__main__":
    main()
