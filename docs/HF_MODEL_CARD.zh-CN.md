---
base_model:
  - microsoft/bitnet-b1.58-2B-4T-bf16
library_name: bit-jev
tags:
  - bitnet
  - structured-decision
  - pointer-head
  - cpu-inference
  - gpu-inference
  - knowledge-distillation
  - yelp
---

# bit-jev-2b-distilled

[English model card](HF_MODEL_CARD.md) · [源码仓库](https://github.com/Zeaulo/bit-jev) · [pip 安装](https://pypi.org/project/bit-jev/)

![bit-jev 训练、蒸馏与 I2_S CPU 推理流程](figures/project-cover.zh-CN.png)

图中三值符号只代表量化 BitLinear 权重；训练与推理是分开的流程。

> 本仓库提供 bit-jev 的 I2_S GGUF 模型包，可通过 `bit-jev` Python 包进行 CPU 或 GPU 推理。权重来自包含 Yelp 评论数据的多源决策训练集。Yelp 权利方许可申请已发出，截至 2026-09-28 尚未收到书面答复。本模型卡公开说明来源与限制；项目代码仓库的 Apache-2.0 许可证不自动适用于此检查点。

## 模型简介

bit-jev-2b-distilled 是一个结构化判断学生模型。输入包含共享 `state` 和一个或多个问题；指针头直接对输入中的候选项打分，返回选择、概率或有序等级。模型不会逐 token 生成自然语言答案。

支持的问题类型：

| 类型 | 输入 | 结构化输出 |
| --- | --- | --- |
| `choice` | 一组选项及说明 | 选择项、分数、概率 |
| `noul` | 是/否问题 | 两类分数与概率 |
| `score` | 有序等级及说明 | 等级分数、期望值、概率 |

## 训练与导出流程

```text
Microsoft BitNet b1.58 2B BF16
        ↓ LoRA 微调 + 指针头训练
bit-jev 初始模型与指针头
        + Kev 9B 教师候选项 logits
完整学生骨干蒸馏 + 指针头
        ↓ I2_S 量化导出
I2_S GGUF + float32 指针头
```

发布包仅含面向原生 CPU runner 的 I2_S 产物、匹配的 tokenizer/config 和头部元数据。训练配置记录了 3,144 步、2 个 epoch、BF16、学习率 `2e-5`、批量 2、梯度累积 4、蒸馏温度 2.0 与权重 1.0。推理头温度在 `pointer.json` 中为 2.35。

## 文件与内存规模

| 文件 | 用途 | 大小 |
| --- | --- | ---: |
| `backbone-i2_s.gguf` | 量化 BitNet 骨干 | 1,187,288,192 字节（约 1.106 GiB） |
| `head.f32` | float32 指针头 | 5,244,948 字节 |
| `pointer.json` | 头部形状、边界 token 与温度 | 770 字节 |
| tokenizer/config 文件 | 请求编码和模型配置 | 约 17.2 MB |

`SHA256SUMS.json` 列出推理包文件的大小和 SHA-256。它不包含自身的哈希。发布包未提供 BF16 safetensors 分片，也未包含训练数据、训练日志或原始评论。

## 推理示例

推荐使用 pip 包。Windows x64 且 CPU 支持 AVX2 时，bit-jev 0.9.10 wheel 已携带 CPU 原生 runner；首次加载会按需下载约 1.19 GB 的模型，推理无需 Git、CMake 或 C++ 编译器。其他系统及 GPU 后端按需从固定源码构建，需要 Git、CMake 3.28+ 和 C++17 编译器；Vulkan 还需要 Vulkan SDK，CUDA 还需要 CUDA Toolkit。

```bash
pip install bit-jev
```

```python
from bit_jev import BitJev

request = {"state": "客户报告重复扣款。", "questions": {"team": {"type": "choice", "instructions": "哪个团队处理？", "criteria": {"billing": "支付退款", "shipping": "物流配送"}}}}
with BitJev.from_pretrained(device="cpu", threads=8) as model:
    print(model.infer(request)["answers"])
```

`device="gpu"` 使用 Vulkan；`device="cuda"` 使用 CUDA 构建。`infer()` 复用常驻模型并返回答案、logits、概率和原生推理耗时。完整 CLI、离线目录和构建细节见[GGUF 安装与推理指南](https://github.com/Zeaulo/bit-jev/blob/main/docs/GGUF_PACKAGE.zh-CN.md)。本模型仓库只保存模型文件；预编译 runner 位于 PyPI 的 Windows x64 wheel。

## 性能案例：AutoDL Xeon Gold 6459C / RTX 5090

以下测量来自一道固定开发题，输入 703 tokens、77 个候选项；推理计时不含加载。CPU 使用 I2_S 原生路径，GPU 使用实验性 FP16 混合精度 PyTorch 路径。这组 GPU 数字不是新 pip 包的 GGUF/Vulkan 或 GGUF/CUDA 测试结果。

| 路径 | 平均推理时间 | 重复次数 | 观测内存 |
| --- | ---: | ---: | ---: |
| CPU，8 线程 | 3,127.88 ms | 3 | 进程峰值 RSS 1,622.74 MiB |
| CPU，16 线程 | 1,972.17 ms | 3 | 进程峰值 RSS 1,624.52 MiB |
| RTX 5090 | 86.56 ms | 5 | GPU 峰值分配 4,935.53 MiB |

![bit-jev AutoDL 单题延迟和内存对比](figures/bit-jev-autodl-case.zh-CN.svg)

[English chart](figures/bit-jev-autodl-case.en.svg)

同一道题上，16 线程 CPU 与 RTX 5090 GPU 路径的延迟比约为 22.8。两条路径使用不同权重格式和数值精度，因此该比值不能解释为纯硬件加速比。CPU RSS 与 GPU 分配量是不同口径。此单题少量重复只作为案例，不代表通用吞吐或准确率承诺。详细的脱敏计时数据见 `benchmark_case_autodl.json`；本包没有收录输入文本、候选内容或预测结果。模型不生成答案 token，因此不适用生成 tokens/s 指标。准确率、Brier、NLL 与 ECE 尚无可复核的公开留出集报告，本页不填入推测值。

## 数据来源与使用边界

训练数据是本地 `decision-v7` 多源决策集，包含 Yelp 评论记录；教师监督来自 Kev 9B 候选项 logits。仓库不包含 Yelp 原始记录。项目已向 Yelp 发送关于衍生权重和指标发布范围的许可申请，截至 2026-09-28 未收到书面答复。下载者应自行审查适用的 Yelp 数据条款及其对衍生权重的影响。

本模型**没有单独授予开放权重许可证**。代码仓库的 Apache-2.0、Microsoft 基础模型许可证及 Kev 源码许可证分别适用于各自作品，不能视为对 Yelp 数据或本衍生检查点的许可。此卡记录项目当前公开状态，不构成法律意见。

## 评测范围与局限

- 当前公开的 bit-jev 结果只有上表的单题延迟与内存案例；微软原版 BitNet 的基础模型基准是另一组独立测量，不能当成本模型成绩。
- 没有提供完整训练/测试来源拆分上的 Accuracy、Brier、NLL、ECE 或置信区间。
- 测试题为开发案例，重复次数少，不能外推到长输入、多问题请求或其他 CPU/GPU。
- 当前原生 CPU runner 每个问题执行一条因果序列；多问题请求会重复处理共享 state。
- 选择结果受候选项措辞、顺序、长度与训练分布影响。重要决策需在目标数据上验证并保留人工复核。

## 引用

```bibtex
@misc{bitjev2026,
  title = {bit-jev-2b-distilled},
  author = {Zeaulo},
  year = {2026},
  url = {https://huggingface.co/jinghao1632/bit-jev-2b-distilled}
}
```

