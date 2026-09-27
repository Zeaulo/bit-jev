# bit-jev

BitNet 骨干 + Kev 风格结构化决策头：把一段共享内容和多个问题直接映射为候选项分数、概率与答案，不生成回答文本。

[English documentation](README.en.md) · [项目总览](versions/project_overall/index.html) · [GitHub Releases](https://github.com/Zeaulo/bit-jev/releases)

![bit-jev 项目流程图](docs/figures/model-highlights.zh-CN.svg)

## 先看结论

| 重点 | 说明 |
| --- | --- |
| 三值骨干 | 量化后的 BitLinear 权重使用 `-1 / 0 / +1`；这不是所有张量都只有三值。 |
| CPU 友好 | I2_S GGUF 压缩骨干配合原生 CPU runner，适合低内存部署。 |
| 非自回归决策 | 指针头直接读取隐藏状态，对选项打分；不需要逐 token 生成答案。 |
| 训练链路 | BitNet BF16 基础模型 → LoRA 微调 → teacher-student 蒸馏 → I2_S 导出。 |
| 任务类型 | `choice` 多选、`noul` 是/否、`score` 有序评分。 |

## 这个项目解决什么问题

通用语言模型擅长生成文本，但很多业务任务只需要在固定选项中做判断。bit-jev 把这类任务改成结构化推理：共享 `state` 只作为上下文输入，调用方声明问题和候选项，模型输出每个问题的答案、选项分数和概率。

模型不会自回归地“蹦”出答案 token。需要注意的是，省去的是答案解码循环，输入仍然要经过 BitNet 骨干的前向计算；当前原生 CPU 路径对多问题请求按问题构造因果序列，因此不能把一次请求简单描述成“一次前向计算”。

## 模型流程

![bit-jev 模型流程与推理边界](docs/figures/model-framework.svg)

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

AutoDL 的 CPU/GPU 单题报告和图表已经在本地整理，但由于它们来自 Yelp 训练检查点，暂不放入公开仓库。许可确认后再补充脱敏图表、模型哈希、完整硬件条件和复现命令；在此之前不要把本地 `test/release/` 草稿当成公开发布物。

## 快速开始

```powershell
git clone https://github.com/Zeaulo/bit-jev.git
Set-Location bit-jev
python -m pip install -e './core'
python test/bootstrap_bitnet.py
python test/bootstrap_bitnet.py --check
```

构建原生程序不需要下载本项目检查点。实际推理需要兼容的 tokenizer/config、I2_S GGUF 骨干和指针头 sidecar；请先确认模型与数据拥有可公开使用的权利。

## 请求格式

```json
{"state":"客户报告同一订单被重复扣款。","questions":{"team":{"type":"choice","instructions":"哪个团队应处理这个问题？","criteria":{"billing":"支付与退款","shipping":"配送问题"}},"escalate":{"type":"noul","instructions":"是否需要升级处理？"}}}
```

一个 UTF-8 JSONL 文件每行一个请求。`state` 在问题之间共享；`questions` 的键由调用方命名。兼容模型返回问题答案、候选项分数、概率和原生计算耗时。

## 文档地图

| 文档 | 适合谁 | 内容 |
| --- | --- | --- |
| [中文 CPU 快速开始](docs/CPU_QUICKSTART.zh-CN.md) | 第一次运行 | 构建、模型文件和 JSONL 调用。 |
| [中文性能规范](docs/BENCHMARKS.zh-CN.md) | 做实验 | 加载、预热、推理、RSS、显存和线程数的统一口径。 |
| [中文模型卡](docs/MODEL_CARD.zh-CN.md) | 评估模型 | 架构、输入契约、局限和发布边界。 |
| [Hugging Face 中文模型卡草稿](docs/HF_MODEL_CARD.zh-CN.md) | 发布模型 | 按 Kev 风格组织训练流程、文件布局、评测切分和发布条件。 |
| [英文 README](README.en.md) | English readers | English overview and reproduction links. |
| [第三方许可](THIRD_PARTY_NOTICES.md) | 发布前 | BitNet、Kev、数据集和检查点的权利边界。 |
| [项目总览网页](versions/project_overall/index.html) | 内部学习 | 代码逻辑、功能需求、数据流和关键目录。 |

## 代码入口

- [`core/bit_jev/api.py`](core/bit_jev/api.py)：请求和答案结构。
- [`core/bit_jev/model.py`](core/bit_jev/model.py)：PyTorch 骨干、分支掩码和指针头。
- [`core/bit_jev/cpu.py`](core/bit_jev/cpu.py)：JSONL 编码与原生进程调用。
- [`core/native/main.cpp`](core/native/main.cpp)：GGUF 骨干和 float32 指针头的 CPU 推理。
- [`core/bit_jev/train.py`](core/bit_jev/train.py)：LoRA 与指针头训练。
- [`core/bit_jev/distill.py`](core/bit_jev/distill.py)：教师 logits 导出和学生蒸馏。
- [`test/`](test/)：构建、基准测试、图表渲染和发布校验脚本。

## 发布状态与许可

仓库代码采用 [Apache-2.0](LICENSE)。BitNet 上游、基础模型、教师模型、Yelp 数据及训练后的检查点各自保留原有条款。Yelp 检查点公开发布需以数据权利方书面许可为依据；许可申请已经发出，正式权重包和派生指标在收到答复前保持待发布状态。

## 引用与致谢

项目实现受 [Kev](https://github.com/jaredpalmer/kev) 的结构化判断接口启发，骨干和原生推理基础来自 [Microsoft BitNet](https://github.com/microsoft/BitNet)。bit-jev 与这些项目的权重、训练结果和许可相互独立。
