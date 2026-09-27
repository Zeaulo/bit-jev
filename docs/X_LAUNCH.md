# v0.11.1 X 首发文案

首帖建议英文并同时附英文项目主视觉 `project-cover.en.png` 与精确 AutoDL 图 `bit-jev-autodl-case.en.png`；随后用仓库内手写客服工单示例解释实际输入形状，再给源码、模型链接与数据来源说明。中文版本单独备选。所有性能数值均对应仓库中的单题脱敏记录，不作为通用性能承诺。

## English launch thread

### Post 1 — attach `figures/project-cover.en.png` and `figures/bit-jev-autodl-case.en.png`

> Most LLM demos make the model talk. bit-jev makes it choose.
>
> One AutoDL dev case: 703 input tokens, 77 options.
> CPU (I2_S, 16 threads): 1.97s, 1.59 GiB RSS.
> RTX 5090 (FP16): 86.6ms, 4.82 GiB allocated.
>
> Different paths/precision; one request, not a general benchmark.

### Reply 1 — concrete included example

> I ran the included support ticket: late shoes, wrong size, double charge.
>
> CPU output: returns 60.8%, shipping 19.6%, billing 19.6%; escalate 55.7%; frustration 1.30/2.
>
> Model scores on a hand-written example, separate from the AutoDL timing case.

### Reply 2 — model and code

> Built on BitNet b1.58: LoRA + pointer head → Kev 9B teacher logits → student distillation → I2_S CPU package.
>
> Code + example: https://github.com/Zeaulo/bit-jev
> Model: https://huggingface.co/jinghao1632/bit-jev-2b-distilled

### Reply 3 — provenance disclosure

> Release note: training included Yelp review records. We requested permission; no written reply as of Sep 28, 2026. The model card grants no standalone open-weights license. The public repo contains no raw reviews.

## 中文备选

### 首帖 — 附 `figures/project-cover.zh-CN.png` 与 `figures/bit-jev-autodl-case.zh-CN.png`

> bit-jev 用指针头直接对候选项打分，不逐 token 生成答案。
>
> AutoDL 单题（703 tokens / 77 选项）：I2_S CPU 16 线程 1.97 秒、RSS 1.59 GiB；RTX 5090 FP16 86.6 毫秒、GPU 分配 4.82 GiB。
>
> 两种推理精度；一道开发题，不代表通用性能。

### 回复 1 — 仓库内的客服工单示例

> 实际运行包内手写客服工单样例：鞋子晚到、尺码错误、重复扣款。CPU 返回退换货 60.8%、物流 19.6%、账单 19.6%；紧急转人工概率 55.7%；不满程度 1.30/2。这是模型输出示例，与上面的 AutoDL 性能测试题不同。

### 回复 2

> 训练与导出：BitNet b1.58 → LoRA + 指针头 → Kev 9B teacher logits → 学生全量蒸馏 → I2_S CPU 模型。
>
> 模型：https://huggingface.co/jinghao1632/bit-jev-2b-distilled
> 代码与脱敏数据：https://github.com/Zeaulo/bit-jev

### 回复 3 — 数据披露

> 训练集包含 Yelp 评论记录。许可申请已发出，截至 2026-09-28 未收到书面答复；模型卡未给权重另行指定开放许可证。公开包不含评论原文或预测结果。

## 发布前检查

- 检查图片中数值与 `docs/benchmark-data/bit-jev-autodl-case-2026-09-27.json` 一致。
- 主视觉只展示概念链路，AutoDL 图才承载具体数值；二者不是同一幅性能图。
- 明确说明 CPU 与 GPU 使用不同权重格式和数值精度，CPU RSS 与 GPU 分配量也不是同一内存口径。
- 确认 GitHub 与 Hugging Face 链接可公开访问，Hub 页面展示中英文模型卡、文件清单和数据状态。
- 不声称有整体准确率、校准成绩、普遍 CPU/GPU 加速比或生成 tokens/s；当前没有可审计的留出集质量报告。
