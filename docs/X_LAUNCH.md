# X launch copy — source-only v0.4.7

These are drafts for the authorized post from the logged-in X account. Publish only after the public repository resolves at [Zeaulo/bit-jev](https://github.com/Zeaulo/bit-jev) and the source-only commit has passed the [release checklist](RELEASE_CHECKLIST.md). This launch shares **code and architecture**, not a trained model or its measurements.

## First post (English)

> Your classifier doesn't need to write an answer.
>
> bit-jev puts Kev-style decisions on BitNet b1.58: state + typed questions in, pointer-head scores out. Native I2_S CPU source is open.
>
> Source only; weights/results await data-rights review.
>
> https://github.com/Zeaulo/bit-jev

Attach the [source architecture PNG](figures/source-architecture.png). Its SVG source is [here](figures/source-architecture.svg). The graphic should show that the native CPU path processes **one row per question**. Do not attach the withheld performance charts or a screenshot of a prediction from the local checkpoint.

## 中文备选首帖

> 分类任务不一定要逐 token 生成答案。
>
> bit-jev 探索把 Kev 式结构化判断放到 BitNet b1.58 上：输入状态和问题，指针头给候选项打分。源码含原生 I2_S CPU 路径。
>
> 本次只发布源码；训练权重和相关结果待数据使用权限审核。
>
> https://github.com/Zeaulo/bit-jev

Use one language for the first post. A translated follow-up can link back to the same source repository. Keep the rights statement in either language.

## Optional follow-ups

Post only when each linked page is public and accurate. Space useful explanations across several days.

1. **Decision interface:** explain `state`, `choice`, `noul`, and `score` with an input-only JSON snippet from the [README](../README.md). Explain that output values require a trained model.
2. **CPU execution path:** show the architecture graphic and explain the boundary: no answer-token decoding, but one native causal row per question and repeated shared-state work for multiple questions.
3. **Reproduction invitation:** point to the [CPU source quick start](CPU_QUICKSTART.md) and [benchmark protocol](BENCHMARKS.md). Invite build feedback and measurements using artifacts and data contributors are permitted to share. State the CPU, thread count, quantization, input length, question count, load inclusion, and memory definition for any future number.

## Launch checks

- Open the repository link while signed out; verify the README, quick start, diagram, license, and source-only tag.
- Inspect the final post text and image in X before posting. Link directly to the repository.
- Do not state that bit-jev weights are downloadable or that accuracy, memory, or speed claims are public. The source-only release has no such public artifact.
- After posting, record the post URL in the release log or launch notes and answer technical questions with the actual source paths and limitations.
