# bit-jev

**安装即用的 BitNet 结构化决策模型。** 给定上下文、问题和候选项，直接返回答案与概率；不用逐 token 生成回答文本。

[![PyPI version](https://img.shields.io/pypi/v/bit-jev?label=PyPI)](https://pypi.org/project/bit-jev/) [![Python 3.11 / 3.12](https://img.shields.io/badge/Python-3.11%20%2F%203.12-3776AB)](https://pypi.org/project/bit-jev/) [![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

[快速开始](#快速开始) · [安装排错](docs/GGUF_PACKAGE.zh-CN.md) · [训练与推理流程](#模型流程) · [实测数据](#速度与内存) · [English](README.en.md) · [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

![bit-jev 三值输入经过决策引擎输出候选结果的浅色主视觉](docs/figures/decision-engine.png)

## 先看结论

| 重点 | 说明 |
| --- | --- |
| 三值骨干 | 量化后的 BitLinear 权重使用 `-1 / 0 / +1`；这不是所有张量都只有三值。 |
| CPU 友好 | I2_S GGUF 压缩骨干配合原生 CPU runner，适合低内存部署。 |
| 非自回归决策 | 指针头直接读取隐藏状态，对选项打分；不需要逐 token 生成答案。 |
| 训练链路 | BitNet BF16 基础模型 → LoRA 微调 → teacher-student 蒸馏 → I2_S 导出。 |
| 任务类型 | `choice` 多选、`noul` 是/否、`score` 有序评分。 |

## 快速开始

在准备运行代码的**同一个 Python 环境**中安装。当前 PyPI 索引可能仍返回旧版；Windows x64、Python 3.11/3.12 可以直接安装[官方 0.11.10 wheel](https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl)：

```bash
python -m pip install --upgrade --no-cache-dir "https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl"
python -c "from importlib.metadata import version; import bit_jev; print(version('bit-jev'), bit_jev.__file__)"
```

第二行应显示发行版版本 `0.11.10` 和当前环境的 `site-packages/bit_jev/__init__.py`。如果显示 `0.4.3` 或导入路径指向另一份源码，请看[安装排错](docs/GGUF_PACKAGE.zh-CN.md)。模型权重不在 wheel 中；第一次调用 `from_pretrained()` 才会下载约 1.19 GB，之后复用缓存。

```python
from bit_jev.gguf import BitJev

request = {
    "state": "客户报告同一订单被重复扣款。",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "哪个团队应处理？",
            "criteria": {"billing": "支付与退款", "shipping": "物流配送"},
        }
    },
}

with BitJev.from_pretrained(device="cpu", threads=8) as model:
    result = model.infer(request)
    print(result["answers"])
    print(result["latency_ms"])
```

Windows x64 且 CPU 支持 AVX2 时，0.11.10 wheel 自带 CPU 与 Vulkan 程序，推理不需要 Git、CMake、编译器或 Vulkan SDK；GPU 仍需兼容的显卡驱动。其他平台及 CUDA 后端会按需从固定源码构建，详见[安装与排错指南](docs/GGUF_PACKAGE.zh-CN.md)。代码采用 Apache-2.0；[模型卡](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)单独说明权重的数据来源和许可状态。

把 `device` 改为 `"gpu"` 可选择 Vulkan；NVIDIA CUDA 专用构建使用 `"cuda"`。多 GPU 主机可传 `gpu_index`。推理接口不会生成 token，`latency_ms` 不含模型加载时间。完整配置、CLI、离线目录与原生构建见 [GGUF 安装与推理指南](docs/GGUF_PACKAGE.zh-CN.md)。

若想直接运行内置题目，也可执行 `python -m bit_jev.demo --threads 8`；此调用不依赖 `bit-jev-demo` 命令是否已经加入 PATH。

## 这个项目解决什么问题

通用语言模型擅长生成文本，但很多业务任务只需要在固定选项中做判断。bit-jev 把这类任务改成结构化推理：共享 `state` 只作为上下文输入，调用方声明问题和候选项，模型输出每个问题的答案、选项分数和概率。

模型不会自回归地“蹦”出答案 token。需要注意的是，省去的是答案解码循环，输入仍然要经过 BitNet 骨干的前向计算；当前原生 CPU 路径对多问题请求按问题构造因果序列，因此不能把一次请求简单描述成“一次前向计算”。

## 模型流程

![bit-jev 中英文分离的训练、蒸馏、量化与推理流程](docs/figures/project-flow.zh-CN.svg)

量化后 **BitLinear 权重**取 `-1 / 0 / +1`，不代表模型所有张量都只有三值。[逐层网络结构图](docs/figures/model-framework.svg)展示掩码、解码层与指针头细节。

1. `train.py` 在 Microsoft BitNet BF16 骨干上训练 LoRA 与指针头。
2. `distill.py export_teacher` 提取教师模型对候选项的 logits。
3. `distill.py train` 使用 teacher logits 训练不带 LoRA 的完整学生骨干。
4. `export_distilled.py` 将学生骨干导出为 I2_S GGUF，并配套 float32 指针头。
5. 原生 runner 读取 JSONL 请求，返回结构化判断结果。

## 速度与内存

公开报告必须同时写清检查点、格式、精度、硬件、输入形状、重复次数和内存口径。仓库保留两类测量：微软公开 BitNet 基础模型的可复现实测，以及本项目训练检查点的 AutoDL 案例数据。两类数据不混在同一张图里。

### 微软公开 BitNet 基础模型

![公开 BitNet 基础模型吞吐](docs/figures/public-base-speed.svg)

![公开 BitNet 基础模型内存](docs/figures/public-base-memory.svg)

这是骨干模型的合成 prefill/decode 测量，不是 bit-jev 分类请求。完整样本和复现命令见[性能测量规范](docs/BENCHMARKS.zh-CN.md)。

### bit-jev AutoDL 单题案例

![bit-jev 单题 AutoDL 延迟与内存对比](docs/figures/bit-jev-autodl-case.zh-CN.svg)

同一台 Xeon Gold 6459C / RTX 5090 主机上测试一道固定开发题，输入 703 tokens、77 个候选项；下表均不计模型加载时间。GPU 测试另排除预热。

| 路径 | 平均推理时间 | 重复次数 | 峰值内存观测 |
| --- | ---: | ---: | ---: |
| CPU，8 线程，I2_S 原生 | 3,127.88 ms | 3 | 进程峰值 RSS 1,622.74 MiB |
| CPU，16 线程，I2_S 原生 | 1,972.17 ms | 3 | 进程峰值 RSS 1,624.52 MiB |
| RTX 5090，FP16 混合精度 | 86.56 ms | 5 | GPU 峰值分配 4,935.53 MiB |

该单题上 GPU 路径约为 16 线程 CPU 路径的 22.8 倍；但两条路径使用不同权重格式和数值精度，不能据此声称纯硬件加速倍数。CPU RSS 与 GPU 分配显存口径不同。原始输入、候选内容和预测值均未公开；逐次计时、模型 SHA-256、实验边界和图表生成脚本见[公开案例数据](docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json)与[性能测量规范](docs/BENCHMARKS.zh-CN.md#bit-jev-autodl-单题案例)。这组小样本不代表通用延迟或准确率。模型权重可从 [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) 下载；使用前请阅读模型卡中的数据来源和许可状态。

源码构建与实验脚本见 [CPU 快速开始](docs/CPU_QUICKSTART.zh-CN.md)。训练、蒸馏和评估依赖可用 `pip install 'bit-jev[train]'` 安装。

## 请求格式

```json
{"state":"客户报告同一订单被重复扣款。","questions":{"team":{"type":"choice","instructions":"哪个团队应处理这个问题？","criteria":{"billing":"支付与退款","shipping":"配送问题"}},"escalate":{"type":"noul","instructions":"是否需要升级处理？"}}}
```

一个 UTF-8 JSONL 文件每行一个请求。`state` 在问题之间共享；`questions` 的键由调用方命名。兼容模型返回问题答案、候选项分数、概率和原生计算耗时。

## 文档地图

| 文档 | 适合谁 | 内容 |
| --- | --- | --- |
| [中文 CPU 快速开始](docs/CPU_QUICKSTART.zh-CN.md) | 第一次运行 | 构建、模型文件和 JSONL 调用。 |
| [GGUF 安装与推理指南](docs/GGUF_PACKAGE.zh-CN.md) | pip 用户 | 按需下载、CPU/GPU 构建、常驻 API 和 CLI。 |
| [中文性能规范](docs/BENCHMARKS.zh-CN.md) | 做实验 | 加载、预热、推理、RSS、显存和线程数的统一口径。 |
| [中文模型卡](docs/MODEL_CARD.zh-CN.md) | 评估模型 | 架构、输入契约、局限和发布边界。 |
| [Hugging Face 中文模型卡](docs/HF_MODEL_CARD.zh-CN.md) | 评估权重 | 训练流程、AutoDL 案例、数据来源、文件哈希和限制。 |
| [英文 README](README.en.md) | English readers | English overview and reproduction links. |
| [第三方许可](THIRD_PARTY_NOTICES.md) | 发布前 | BitNet、Kev、数据集和检查点的权利边界。 |
| [项目总览网页](versions/project_overall/index.html) | 内部学习 | 代码逻辑、功能需求、数据流和关键目录。 |

## 代码入口

- [`core/bit_jev/api.py`](core/bit_jev/api.py)：请求和答案结构。
- [`core/bit_jev/model.py`](core/bit_jev/model.py)：PyTorch 骨干、分支掩码和指针头。
- [`core/bit_jev/cpu.py`](core/bit_jev/cpu.py)：JSONL 编码与原生进程调用。
- [`core/bit_jev/encoding.py`](core/bit_jev/encoding.py)：不加载 PyTorch 的请求编码。
- [`core/bit_jev/gguf.py`](core/bit_jev/gguf.py)：模型下载与常驻 Python API。
- [`core/bit_jev/native_build.py`](core/bit_jev/native_build.py)：固定上游版本的按需 CPU/GPU 构建。
- [`core/native/main.cpp`](core/native/main.cpp)：GGUF 骨干和 float32 指针头的 CPU 推理。
- [`core/bit_jev/train.py`](core/bit_jev/train.py)：LoRA 与指针头训练。
- [`core/bit_jev/distill.py`](core/bit_jev/distill.py)：教师 logits 导出和学生蒸馏。
- [`test/`](test/)：构建、基准测试、图表渲染和发布校验脚本。

## 发布状态与许可

仓库代码采用 [Apache-2.0](LICENSE)。I2_S 检查点与脱敏 AutoDL 派生指标现已发布到 [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)。训练集包含 Yelp 评论记录；许可申请已发出，截至 2026-09-28 尚未收到书面答复。本项目没有为权重另行指定开放许可证，代码仓库许可证不自动覆盖权重或第三方数据。公开案例只含聚合计时与内存，不含评论原文、候选内容或模型预测。

## 引用与致谢

项目实现受 [Kev](https://github.com/jaredpalmer/kev) 的结构化判断接口启发，骨干和原生推理基础来自 [Microsoft BitNet](https://github.com/microsoft/BitNet)。bit-jev 与这些项目的权重、训练结果和许可相互独立。
