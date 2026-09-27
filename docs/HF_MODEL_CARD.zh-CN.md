# bit-jev-2b-distilled

> Hugging Face 发布草稿。当前版本只用于本地审阅；Yelp 训练数据的权利方书面答复尚未收到，因此检查点和派生指标尚未上传。

## 模型简介

bit-jev-2b-distilled 是一个用于结构化决策的 BitNet 学生模型。它接受一段共享 `state` 和一个或多个问题，在调用方给出的候选项上返回分数、概率和答案。模型输出结构化决策，不逐 token 生成回答文本。

训练链路：

```text
Microsoft BitNet BF16 backbone
        ↓
LoRA + pointer head fine-tuning
        ↓ export_teacher
teacher logits
        ↓ distill.py train
full student backbone + pointer head
        ↓ export_distilled.py
I2_S GGUF + float32 pointer head
```

## 任务类型

| 类型 | 输入 | 输出 |
| --- | --- | --- |
| `choice` | 候选项名称及说明 | 选择项、概率、置信度 |
| `noul` | 是/否问题 | `noul` 概率 |
| `score` | 有序等级列表 | 期望等级、概率、置信度 |

一个请求可以同时包含三种问题。问题共享 `state`，但不能读取其他问题的文本。

## 训练和蒸馏配置

| 项目 | 记录 |
| --- | --- |
| 基础模型 | `microsoft/bitnet-b1.58-2B-4T-bf16` |
| 初始学生 | LoRA 微调后的 `runs/bit-jev-2b` |
| 教师输出 | `runs/distill/kev9b_logits.jsonl` |
| 蒸馏轮数 | 2 |
| 学习率 | `2e-5` |
| 批量 / 梯度累积 | 2 / 4 |
| 蒸馏温度 | 2.0 |
| 蒸馏权重 | 1.0 |
| 训练步数 | 3,144 |
| 学生精度 | BF16 训练，I2_S 导出 |

完整训练数据清单、拆分方式和许可依据应在获得数据权利方书面确认后补充。仓库不发布原始 Yelp 评论记录。

## 文件布局

最终 Hub 仓库应至少包含：

- `backbone-i2_s.gguf`：原生 CPU 推理骨干。
- `head.f32`：float32 指针头 sidecar。
- `pointer.json`：边界标记、头部布局和温度元数据。
- `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`：输入编码文件。
- `config.json`：学生骨干配置。
- `README.md`：本模型卡的发布版。
- `SHA256SUMS.json`：逐文件 SHA-256 清单。

I2_S GGUF 的本地候选文件为 1,187,288,192 字节；正式发布前需要在干净环境重新打包并核对摘要。完整 BF16 分片不作为 CPU 包的替代品上传，除非模型卡明确说明它们的用途、尺寸和许可。

## 本地 AutoDL 案例（未发布）

同一 Xeon Gold 6459C / RTX 5090 主机上的单题案例包含 703 个输入 token 和 77 个候选项，推理时间不含加载。CPU I2_S 原生路径与 GPU FP16 混合精度路径使用不同格式，不能把延迟比解释为纯硬件加速比。逐次样本与脱敏候选稿保存在项目根目录的 `test/release/`，收到书面许可后再转成公开表格和图。

## 使用方式

公开版应提供一个与仓库 CPU runner 对齐的最小示例：

```powershell
python -m pip install -e './core'
python test/bootstrap_bitnet.py --check
python -m bit_jev.cpu --model ./backbone-i2_s.gguf --head ./head.f32 --input ./example_cpu_request.jsonl
```

具体参数以发布时的 CPU quick start 为准。模型不会把选项答案逐 token 解码出来；性能报告应使用每请求延迟、每问题延迟、输入 token/s 和峰值 RSS，而不是生成 token/s。

## 评测报告结构

发布版应仿照 Kev 的模型卡把结果分开：

1. 已训练来源（训练数据分布内的留出集）。
2. 新来源（训练阶段未出现的数据或规则）。
3. 每个问题类型的样本数、准确率、Brier、NLL、ECE 和温度校准结果。
4. CPU 与 GPU 的硬件、线程、量化格式、精度、预热、同步方式和内存定义。
5. 每个公开检查点的版本、文件哈希和基准输入形状。

不要把开发集、测试集和单题 smoke test 的数字放在同一张“总准确率”图里。

## 许可与发布状态

代码仓库采用 Apache-2.0。BitNet 基础模型、上游实现、教师模型和 Yelp 数据各自适用其原有条款。向 Hugging Face 公开上传本检查点及其派生指标前，需要保存覆盖权重下载、量化导出、指标展示和下游使用范围的书面许可。

