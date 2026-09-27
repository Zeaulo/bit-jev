"""把同一份 I2_S GGUF 的三元权重展开成 F16 CPU 对照文件。"""

import argparse
import sys
from pathlib import Path

import numpy as np


# 使用仓库内的 GGUF 读写实现，保持原模型元数据与张量顺序。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learning/bitnet/3rdparty/llama.cpp/gguf-py"))
import gguf  # noqa: E402


def unpack_i2s(tensor):
    """按 128 个三元值交错打包协议恢复一个权重矩阵。"""
    count = int(np.prod(tensor.shape))
    if count % 128:
        raise ValueError(f"I2_S 张量元素数不是 128 的整数倍：{tensor.name}")
    packed_bytes = count // 4
    raw = tensor.data.reshape(-1)
    if len(raw) != packed_bytes + 32:
        raise ValueError(f"I2_S 张量大小异常：{tensor.name}")
    scale = np.frombuffer(raw[packed_bytes:packed_bytes + 4].tobytes(), dtype="<f4")[0]
    packed = raw[:packed_bytes].reshape(-1, 32)
    # 每个字节的高到低四组 2-bit 值依次属于块内四段 32 元素。
    quarters = np.stack(((packed >> 6) & 3, (packed >> 4) & 3,
                         (packed >> 2) & 3, packed & 3), axis=1)
    ternary = quarters.reshape(-1).astype(np.int8) - 1
    if np.any((ternary < -1) | (ternary > 1)):
        raise ValueError(f"I2_S 张量包含非三元编码：{tensor.name}")
    values = (ternary.astype(np.float32) * scale).astype(np.float16)
    return values.reshape(tuple(reversed(tensor.shape)))


def convert(source_path, output_path):
    """复制配置和非 I2_S 张量，逐矩阵反量化 I2_S 权重。"""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"F16 对照文件已存在：{output_path}")
    reader = gguf.GGUFReader(source_path)
    architecture = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(output_path, architecture)
    # 架构与文件类型由 writer 重新写入，其余训练和 tokenizer 元数据照原样保留。
    for name, field in reader.fields.items():
        if name in ("general.architecture", "general.file_type"):
            continue
        field_type = field.types[0]
        sub_type = field.types[-1] if field_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), field_type, sub_type)
    writer.add_file_type(1)
    changed = 0
    for tensor in reader.tensors:
        if tensor.tensor_type == gguf.GGMLQuantizationType.I2_S:
            writer.add_tensor(tensor.name, unpack_i2s(tensor))
            changed += 1
        else:
            writer.add_tensor(tensor.name, tensor.data,
                              raw_shape=tuple(reversed(tensor.shape)),
                              raw_dtype=tensor.tensor_type)
    if changed == 0:
        raise ValueError("输入 GGUF 没有 I2_S 张量")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"已转换 {changed} 个 I2_S 张量：{output_path}", flush=True)


def main():
    """读取输入路径并创建不可覆盖的 F16 对照文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    convert(args.input, args.out)


if __name__ == "__main__":
    main()
