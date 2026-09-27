"""BitNet 结构化决策：训练、I2_S 导出与逐题 CPU 推理共用同一编码契约。"""

# 包版本与 pyproject.toml、项目更新日志保持同步。
__version__ = "0.3.3"

# 默认骨干用于新训练运行；蒸馏和 CPU 推理从检查点读取实际来源。
DEFAULT_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"
