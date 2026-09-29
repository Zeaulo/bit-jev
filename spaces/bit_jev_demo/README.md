---
title: bit-jev · 中英双语 CPU 在线测试
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# bit-jev · 在线测试 / Live demo

中文和英文页面提供 Choice（选择题）、Noul（是非题）、Score（等级题）三个 Tab。Choice 默认两个选项，Score 默认三个等级，均可添加、删除，保留至少两项且最多四项。背景与每项说明可以留空；背景会传入“常见问题”，说明会复制对应选项或等级。页面先显示结论与候选概率条，完整 JSON 收在高级结果中。页面调用同一公开 I2_S GGUF 检查点；首次请求会下载约 1.19 GB 权重并加载，显示的原生计算耗时不含下载和加载。

The Chinese and English forms provide Choice, Noul, and Score tabs. Choice starts with two options and Score with three levels. Both allow adding and removing items, with two to four active items. Background and item descriptions are optional. The forms show the answer and probability bars first, with full JSON in an advanced section. They call the same public I2_S GGUF checkpoint. The first request downloads and loads about 1.19 GB; download and loading are excluded from the displayed native compute time.

- [GitHub 源码 / source](https://github.com/Zeaulo/bit-jev)
- [Hugging Face 模型 / model](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)
- [ModelScope 模型 / model](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled)

页面使用免费的 2 vCPU CPU 环境；这里的运行时间不等于仓库中 32 核 EPYC 和 RTX 5090 的测量。模型训练数据包含 Yelp 评论；模型卡说明数据来源与目前的许可状态。

The demo uses a free 2-vCPU CPU environment. Its timing cannot be compared directly with the repository's 32-core EPYC or RTX 5090 measurements. The checkpoint was trained partly on Yelp review data; consult the model card for provenance and licensing status.
