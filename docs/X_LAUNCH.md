# v0.10.0 X 首发文案

首帖建议英文并附英文 AutoDL 图，之后接两条回复：一条解释模型链路并放下载链接，一条披露 Yelp 数据和权重许可状态。中文版本单独备选。所有数值均对应仓库中的单题脱敏记录，不作为通用性能承诺。

## English launch thread

### Post 1 — attach `figures/bit-jev-autodl-case.en.png`

> bit-jev scores supplied options instead of generating answer tokens.
>
> One AutoDL case (703 input tokens, 77 options):
> CPU · I2_S · 16 threads: 1.97s / 1.59 GiB RSS
> RTX 5090 · FP16: 86.6ms / 4.82 GiB GPU allocation
>
> Different paths, one development request—not a general benchmark.

### Reply 1

> BitNet b1.58 → LoRA + pointer head → Kev 9B teacher logits → full-student distillation → I2_S CPU package.
>
> Model: https://huggingface.co/jinghao1632/bit-jev-2b-distilled
> Code + sanitized samples: https://github.com/Zeaulo/bit-jev

### Reply 2 — provenance disclosure

> Disclosure: training included Yelp review records. The permission request has no written reply as of Sep 28, 2026; the model card specifies no standalone open-weights license. No raw reviews or predictions are distributed.

## 中文备选

### 首帖 — 附 `figures/bit-jev-autodl-case.zh-CN.png`

> bit-jev 用指针头直接对候选项打分，不逐 token 生成答案。
>
> AutoDL 单题（703 tokens / 77 选项）：I2_S CPU 16 线程 1.97 秒、RSS 1.59 GiB；RTX 5090 FP16 86.6 毫秒、GPU 分配 4.82 GiB。
>
> 两种推理精度；一道开发题，不代表通用性能。

### 回复 1

> 训练与导出：BitNet b1.58 → LoRA + 指针头 → Kev 9B teacher logits → 学生全量蒸馏 → I2_S CPU 模型。
>
> 模型：https://huggingface.co/jinghao1632/bit-jev-2b-distilled
> 代码与脱敏数据：https://github.com/Zeaulo/bit-jev

### 回复 2 — 数据披露

> 训练集包含 Yelp 评论记录。许可申请已发出，截至 2026-09-28 未收到书面答复；模型卡未给权重另行指定开放许可证。公开包不含评论原文或预测结果。

## 发布前检查

- 检查图片中数值与 `docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json` 一致。
- 明确说明 CPU 与 GPU 使用不同权重格式和数值精度，CPU RSS 与 GPU 分配量也不是同一内存口径。
- 确认 GitHub 与 Hugging Face 链接可公开访问，Hub 页面展示中英文模型卡、文件清单和数据状态。
- 不声称有整体准确率、校准成绩、普遍 CPU/GPU 加速比或生成 tokens/s；当前没有可审计的留出集质量报告。
