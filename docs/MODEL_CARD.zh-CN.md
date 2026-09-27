# bit-jev：架构与模型文件状态

[English](MODEL_CARD.md) · [返回中文 README](../README.md)

## 发布状态

**bit-jev-2b-distilled I2_S 模型包和脱敏 AutoDL 单题指标现已发布在 [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)。** 训练集包含 Yelp 评论记录；许可申请已发出，截至 2026-09-28 尚未收到书面答复。模型卡明确记录该状态，且没有为权重单独指定开放许可证。另行发布的[微软公开基础模型独立测量](BENCHMARKS.zh-CN.md#微软公开-bitnet-基础模型独立实测)只测原版骨干执行，不代表 bit-jev 分类。

本页说明代码结构；训练流程、模型文件、单题性能案例和限制见[中英文 Hugging Face 模型卡](HF_MODEL_CARD.zh-CN.md)。

## 架构

项目的结构化判断接口受 [Kev](https://github.com/jaredpalmer/kev) 启发，骨干设计采用 [Microsoft BitNet b1.58 2B 系列](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)。一个请求提供共享 `state` 与一个或多个问题：

| 类型 | 决策形式 |
| --- | --- |
| `choice` | 在调用者给出的候选项中选择。 |
| `noul` | 判断是或否。 |
| `score` | 在有序等级上评分。 |

指针头读取候选项边界和决策位置的隐藏状态，计算选项 logits 与概率，不生成回答文本。PyTorch 打包路径用分块因果注意力掩码隔离问题分支，并共享 `state` 的计算。当前原生 CPU 路径则**每个问题运行一条因果序列**，多问题请求会重复处理共享内容。

首页[中文核心图](figures/model-highlights.zh-CN.svg)按源码展示训练到部署的先后顺序：`train.py` 在 BitNet BF16 基础模型上训练 LoRA 与指针头；`distill.py export_teacher` 从已训练模型提取候选项 logits；`distill.py train` 使用这些目标训练不带 LoRA 的完整学生骨干，并使用微调模型的指针头；`export_distilled.py` 将兼容学生骨干导出为 I2_S GGUF。图中的三值是**量化后的 BitLinear 权重**，不适用于所有张量。图上的 CPU 数字只来自微软公开基础模型独立实测，不能当作此训练链路的最终成绩。

原生 CPU 程序读取、输出 JSONL；它不是 HTTP 服务。`core/bit_jev/serve.py` 的 PyTorch 服务属于另一条执行路径，不能把两者的计时与内存混为一谈。

## 已发布的 CPU 模型包

| 文件或组件 | 用途 |
| --- | --- |
| 匹配的分词器与配置 | 用正确的专用标记、模型维度编码请求。 |
| I2_S GGUF 骨干 | 由原生 CPU 程序运行量化后的 BitNet 骨干。 |
| Float32 指针头 sidecar | 对候选项和决策位置的隐藏状态评分。 |
| 指针头元数据 | 记录边界标记 ID、头部布局和温度参数。 |
| 来源与文件哈希 | Hugging Face 模型卡披露基础模型、训练数据状态、导出步骤与逐文件 SHA-256。 |

[CPU 构建与使用](CPU_QUICKSTART.zh-CN.md)展示如何编译程序、下载 Hub 文件并调用模型。

## 预期用途与限制

代码适合研究显式候选项的路由、是/否决策和有序评分；它不是通用聊天生成器。已公开的 AutoDL 数值来自一道开发题，不能代表准确率或通用速度。速度与内存由具体权重、编译方式、CPU、问题数、输入长度和候选项数量共同决定。

将任何自行训练的模型用于业务前，应在有权使用且符合目标场景的数据上评估。检查候选项顺序对结果的影响、CPU 与参考路径的数值一致性、概率校准和错误代价。对于重要决策，在对应场景完成验证之前应保留人工复核。

## 已公开评测及许可证说明

已公开测量记录了硬件、输入形状、重复次数、量化/精度与内存定义；目前没有可以审计的完整留出集 Accuracy、Brier、NLL 或 ECE 报告。[性能测量规范](BENCHMARKS.zh-CN.md)说明加载、预热、常驻推理和内存口径，并链接逐次脱敏数据。

本仓库代码采用 [Apache-2.0](../LICENSE)。上游源码与基础模型的说明见[第三方许可说明](../THIRD_PARTY_NOTICES.md)。代码许可证不自动适用于 Hugging Face 权重，也不解决 Yelp 数据训练所得检查点的分发许可。
