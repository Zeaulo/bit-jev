"""验证蒸馏导出的 GGUF 文件头、张量目录及决策头契约。"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch


# 项目根目录确定参考仓库中 GGUFReader 的位置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GGUF_PACKAGE = PROJECT_ROOT / "learning/bitnet/3rdparty/llama.cpp/gguf-py"
sys.path.insert(0, str(GGUF_PACKAGE))


def digest(path):
    """以固定块大小重新计算来源文件摘要。"""
    # 摘要对象累积各文件块。
    result = hashlib.sha256()
    # 文件句柄在计算结束后自动关闭。
    with path.open("rb") as source:
        # 循环读取文件块，控制峰值内存。
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify(run, artifact):
    """逐项核验导出来源、GGUF 结构和指针头元数据。"""
    # 清单记录来源文件及 GGUF 文件名。
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    # 指针协议保存分类计算所需的独立信息。
    pointer = json.loads((artifact / "pointer.json").read_text(encoding="utf-8"))
    # GGUF 文件由清单指定，避免凭文件名猜测。
    gguf_path = artifact / manifest["gguf_file"]
    if gguf_path.stat().st_size != manifest["gguf_bytes"]:
        raise AssertionError("GGUF 文件大小与清单不一致")
    # 三项小型文件摘要防止把不同训练运行的文件混用。
    for filename, key in (("config.json", "config_sha256"),
                          ("model.safetensors.index.json", "index_sha256"),
                          ("head.pt", "head_sha256")):
        if digest(run / filename) != manifest[key]:
            raise AssertionError(f"来源文件摘要不匹配：{filename}")
    if digest(artifact / "head.pt") != manifest["head_sha256"]:
        raise AssertionError("导出目录中的决策头与来源不一致")
    # 权重分片大小用于排查曾经出现过的下载截断问题。
    for filename, expected_size in manifest["source_shards"].items():
        if (run / filename).stat().st_size != expected_size:
            raise AssertionError(f"来源分片大小变化：{filename}")

    # GGUF v2/v3 文件头含版本、张量数和元数据项数。
    with gguf_path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        raise AssertionError("GGUF 文件头无效")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:])
    if version not in (2, 3) or tensor_count < 200 or metadata_count == 0:
        raise AssertionError("GGUF 版本、张量数或元数据数量异常")
    # 仓库自带读取器会解析全部张量目录，并在此发现偏移或类型错误。
    import gguf
    reader = gguf.GGUFReader(gguf_path)
    if len(reader.tensors) != tensor_count:
        raise AssertionError("GGUF 目录张量数与文件头不一致")
    # 将实际 GGUF 名称与原始索引逐一对应，避免仅凭数量相等放过漏写或重名。
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    index = json.loads((run / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET_B158, config["num_hidden_layers"])
    expected_names = set()
    for source_name in index["weight_map"]:
        mapped = mapping.get_name(key=source_name, try_suffixes=(".weight", ".bias"))
        if mapped is None and source_name.startswith("layers."):
            mapped = mapping.get_name(key="model." + source_name, try_suffixes=(".weight", ".bias"))
        if mapped is None:
            raise AssertionError(f"无法映射来源张量：{source_name}")
        expected_names.add(mapped)
    actual_names = {tensor.name for tensor in reader.tensors}
    if actual_names != expected_names:
        raise AssertionError(f"GGUF 张量名称不一致：缺少 {sorted(expected_names - actual_names)[:5]}，多出 {sorted(actual_names - expected_names)[:5]}")
    # 实际导出必须至少包含一个 I2_S 权重张量。
    quantized = sum(t.tensor_type == gguf.GGMLQuantizationType.I2_S for t in reader.tensors)
    if quantized == 0:
        raise AssertionError("GGUF 中未找到 I2_S 张量")

    # 决策头维度和温度必须与原始 PyTorch 检查点一致。
    head = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    if pointer["head_dim"] != head["head_dim"] or pointer["temperature"] != head["temperature"]:
        raise AssertionError("决策头维度或温度不一致")
    # 五个分隔符必须有互不重复的 token ID。
    delimiter_ids = [item["id"] for item in pointer["delimiters"].values()]
    if len(delimiter_ids) != 5 or len(set(delimiter_ids)) != 5:
        raise AssertionError("分隔符 ID 不完整或重复")
    # GGUF 内嵌词表也必须在这些 ID 上保留相同控制 token。
    gguf_tokens = reader.fields["tokenizer.ggml.tokens"]
    for delimiter in pointer["delimiters"].values():
        if gguf_tokens.contents(delimiter["id"]) != delimiter["token"]:
            raise AssertionError(f"GGUF 词表中的分隔符不一致：{delimiter['token']}")
    print(f"GGUF 校验通过：版本 {version}，{tensor_count} 个张量，{quantized} 个 I2_S 张量")


def main():
    """读取命令行目录参数并输出一次校验结果。"""
    # 参数均为显式路径，避免误用默认训练运行。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    verify(Path(args.run).resolve(), Path(args.artifact).resolve())


if __name__ == "__main__":
    main()
