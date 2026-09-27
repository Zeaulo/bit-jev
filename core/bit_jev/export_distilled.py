"""将全参数蒸馏检查点导出为 bitnet.cpp 的 I2_S GGUF 与决策头元数据。"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open

from .checkpoint import BitJevCheckpoint
from .cpu_head import write_head_binary
from .model import SPECIAL_TOKENS, delimiter_ids, load_tokenizer


# 项目根目录用于定位参考转换器及其 llama.cpp 子模块。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERTER = PROJECT_ROOT / "learning/bitnet/utils/convert-hf-to-gguf-bitnet.py"
GGUF_PACKAGE = PROJECT_ROOT / "learning/bitnet/3rdparty/llama.cpp/gguf-py"


def file_digest(path):
    """按块计算小型元数据文件的 SHA-256，避免把文件一次读入内存。"""
    # 哈希对象保存已处理字节的累积摘要。
    digest = hashlib.sha256()
    # 文件句柄限定在函数内部关闭。
    with path.open("rb") as source:
        # 每轮读取 1 MiB，直到文件末尾。
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_source(run, converter):
    """在创建输出目录前检查蒸馏权重、配置、转换器和依赖。"""
    # 输入目录必须是本地完整检查点，避免导出时意外联网。
    if not run.is_dir():
        raise FileNotFoundError(f"蒸馏模型目录不存在：{run}")
    # 配置文件决定转换器采用在线量化路径。
    config_path = run / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少模型配置：{config_path}")
    # 模型配置必须匹配当前导出算法。
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "bitnet" or quant.get("quantization_mode") != "online":
        raise ValueError("只支持在线量化的 BitNet 蒸馏检查点")
    # LoRA 检查点需要另一条导出路径，不能误当成全参数蒸馏模型。
    if (run / "adapter_model.safetensors").exists():
        raise ValueError("输入仍包含 LoRA adapter，不能按全参数蒸馏模型导出")
    # 权重索引列出所有必需分片，逐个检查其存在和大小。
    index_path = run / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"缺少权重索引：{index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("权重索引没有张量映射")
    # 每个分片都必须非空，防止沿用此前中断的下载。
    for shard_name in shard_names:
        shard = run / shard_name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"权重分片缺失或为空：{shard}")
        # safe_open 会检查 safetensors 头部与实际文件边界，识别非空但截断的分片。
        with safe_open(shard, framework="pt", device="cpu") as source:
            if not source.keys():
                raise ValueError(f"权重分片没有张量：{shard}")
    # 决策头和 tokenizer 是生成 CPU 侧接口契约的必要输入。
    for required in ("head.pt", "tokenizer.json", "tokenizer_config.json"):
        if not (run / required).is_file():
            raise FileNotFoundError(f"缺少导出输入：{run / required}")
    # 转换器依赖仓库中的 I2_S 定义，普通 gguf 包未必兼容。
    if not converter.is_file():
        raise FileNotFoundError(f"缺少 BitNet 转换器：{converter}")
    if not (GGUF_PACKAGE / "gguf").is_dir():
        raise FileNotFoundError("缺少 llama.cpp/gguf-py 子模块，请初始化 learning/bitnet/3rdparty/llama.cpp")
    return config_path, index_path, shard_names


def make_converter_view(run, staging, shard_names):
    """构造兼容转换器的临时 HF 目录，不复制或改写大型权重分片。"""
    # 蒸馏训练保存的是裸 BitNetModel；转换器只注册了 ForCausalLM 名称。
    source_view = staging / "converter-input"
    source_view.mkdir()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = ["BitNetForCausalLM"]
    (source_view / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # tokenizer 及索引文件体积较小，拷贝后可由转换器原样读取。
    for filename in ("model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        source = run / filename
        if source.is_file():
            shutil.copy2(source, source_view / filename)
    # 同盘硬链接指向已验证的权重分片，不消耗第二份约 9.6 GB 空间。
    for filename in shard_names:
        os.link(run / filename, source_view / filename)
    return source_view


def remove_converter_view(source_view):
    """仅移除导出程序自己创建的浅层文件及空目录。"""
    # 临时输入目录应直接位于本次 staging 下，避免误删其他目录。
    if source_view.name != "converter-input" or not source_view.parent.name.endswith(".partial"):
        raise ValueError(f"拒绝清理非临时转换目录：{source_view}")
    # 此目录只有普通文件和硬链接，不递归清理。
    for source_file in source_view.iterdir():
        if not source_file.is_file():
            raise ValueError(f"临时输入目录出现意外子目录：{source_file}")
        source_file.unlink()
    source_view.rmdir()


def export(run, out, converter=DEFAULT_CONVERTER):
    """转换 backbone，保存指针头及接口元数据，并原子发布输出目录。"""
    # 所有路径先解析成绝对路径，保证从任意工作目录调用结果一致。
    run = Path(run).resolve()
    out = Path(out).resolve()
    converter = Path(converter).resolve()
    config_path, index_path, shard_names = check_source(run, converter)
    # 已有正式或临时目录都需要人工处理，避免覆盖既有模型。
    staging = out.with_name(out.name + ".partial")
    if out.exists() or staging.exists():
        raise FileExistsError(f"输出目录已存在：{out if out.exists() else staging}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()

    # 只为转换器子进程加入仓库自带的 GGUF Python 实现。
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(GGUF_PACKAGE), environment.get("PYTHONPATH", "")) if part
    )
    source_view = make_converter_view(run, staging, shard_names)
    gguf_path = staging / "backbone-i2_s.gguf"
    command = [sys.executable, str(converter), str(source_view), "--outtype", "i2_s",
               "--outfile", str(gguf_path)]
    # 子进程失败时保留 .partial 供诊断，但始终移除其中的权重硬链接。
    try:
        subprocess.run(command, check=True, env=environment)
    finally:
        remove_converter_view(source_view)
    # GGUF 头必须含实际张量；转换器旧版曾把零张量文件误报为成功。
    with gguf_path.open("rb") as gguf_file:
        header = gguf_file.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        raise ValueError("转换结果缺少 GGUF 文件头")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:])
    if version not in (2, 3) or tensor_count < 200 or metadata_count == 0:
        raise ValueError(f"GGUF 张量目录异常：版本 {version}，张量 {tensor_count}，元数据 {metadata_count}")

    # 原始 head.pt 保留完整 PyTorch 状态字典，便于后续 CPU runner 读取。
    shutil.copy2(run / "head.pt", staging / "head.pt")
    checkpoint = BitJevCheckpoint(run)
    tokenizer = load_tokenizer(run)
    token_ids = delimiter_ids(tokenizer)
    # 指针协议记录当前原生 CPU 逐题因果行与独立决策头的执行方式。
    pointer = {
        "arch": checkpoint.meta.arch,
        "head_dim": checkpoint.meta.head_dim,
        "temperature": checkpoint.meta.temperature,
        "delimiters": dict(zip(("state", "question", "option_open", "option_close", "decide"),
                               ({"token": token, "id": token_id}
                                for token, token_id in zip(SPECIAL_TOKENS, token_ids)))),
        "readout": "logits_k = dot(Wq h_decide / sqrt(d), Wk h_option_close_k) / temperature",
        "cpu_runner": "one causal row per question; native hidden-state extraction and float32 pointer head",
    }
    (staging / "pointer.json").write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 清单记录输入分片与关键元数据的指纹，便于后续追溯导出来源。
    manifest = {
        "format": "bit-jev-distilled-i2_s-v1",
        "source_run": str(run),
        "source_shards": {name: (run / name).stat().st_size for name in shard_names},
        "config_sha256": file_digest(config_path),
        "index_sha256": file_digest(index_path),
        "head_sha256": file_digest(run / "head.pt"),
        "gguf_bytes": gguf_path.stat().st_size,
        "gguf_tensor_count": tensor_count,
        "gguf_file": gguf_path.name,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 原生 CPU runner 读取定长 float32 sidecar，无需解析 PyTorch 序列化格式。
    write_head_binary(run, staging)
    staging.rename(out)
    return out


def main(argv=None):
    """解析命令行参数并执行一次不可覆盖的导出。"""
    # 参数解析器向本地工作流暴露输入、输出和转换器位置。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="全参数蒸馏检查点目录")
    parser.add_argument("--out", required=True, help="新建的导出目录")
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER), help="BitNet GGUF 转换器")
    args = parser.parse_args(argv)
    result = export(args.run, args.out, args.converter)
    print(f"导出完成：{result}")


if __name__ == "__main__":
    main()
