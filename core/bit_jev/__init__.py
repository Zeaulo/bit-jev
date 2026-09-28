"""BitNet 结构化决策：训练、I2_S 导出与逐题 CPU 推理共用同一编码契约。"""

# 运行时版本只从单独的版本文件读取，避免安装包与原生缓存使用不同版本号。
from .__version__ import __version__

# 默认骨干用于新训练运行；蒸馏和 CPU 推理从检查点读取实际来源。
DEFAULT_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"

# 顶层 API 延迟导入，单纯查询包版本时不会初始化 tokenizer 或原生构建器。
__all__ = ["BitJev", "download_model", "__version__"]


def __getattr__(name):
    """按需公开 GGUF 常驻模型与模型下载入口。"""
    if name in {"BitJev", "download_model"}:
        from .gguf import BitJev, download_model
        return {"BitJev": BitJev, "download_model": download_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
