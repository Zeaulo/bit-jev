# X launch copy — source, model framework and public-base benchmark v0.6.7

These are drafts for the authorized post from the logged-in X account. Publish after the updated [Zeaulo/bit-jev](https://github.com/Zeaulo/bit-jev) repository displays the source and the independently measured public-base charts. The launch shares **source and measurements of Microsoft's public BitNet backbone**, not a trained bit-jev checkpoint or its performance.

## First post (English; attach speed chart)

> On my laptop, CPU beat the Vulkan hybrid path for Microsoft's public BitNet I2_S: 56.2 vs 45.2 input tok/s (Ryzen 7 4800H / RTX 2060; 5 runs; load excluded). GPU path still uses CPU RAM.
>
> Raw data + source: https://github.com/Zeaulo/bit-jev

Attach the [public-base speed PNG](figures/public-base-speed.png). The image itself states that this is a synthetic base-model backbone benchmark, not bit-jev classification.

## First reply (English)

> bit-jev = BitNet backbone + Kev-style structured decisions. It scores given options without decoding answer text token by token. The repo publishes source only; the trained bit-jev checkpoint and its performance remain withheld pending data-rights review.
>
> Method + samples: https://github.com/Zeaulo/bit-jev/blob/main/docs/BENCHMARKS.md

Attach the [neural model framework PNG](figures/model-framework.png) to this reply. It shows the branch mask, decoder layer and pointer readout and marks the native CPU row path separately.

## 中文备选首帖

> 笔记本 CPU 跑 BitNet，有时能比这台机器的 GPU 混合路径更快。
>
> 同一份微软公开 I2_S GGUF：Ryzen 7 4800H 预填充 56.2 token/s，RTX 2060 Vulkan 混合路径 45.2。各 5 次，计时不含模型加载；GPU 路径仍使用 CPU 内存，不能泛化为“CPU 比 GPU 强”。
>
> bit-jev 把 BitNet 骨干接到 Kev 式候选评分接口。源码已开源；bit-jev 训练权重和其跑分暂不公开。
>
> https://github.com/Zeaulo/bit-jev

Use one language for the first post. A translated follow-up can link back to the same source repository. Keep the distinction between public BitNet and withheld bit-jev results in either language.

## Optional follow-ups

Post only when each linked page is public and accurate. Space useful explanations across several days.

1. **Decision interface:** explain `state`, `choice`, `noul`, and `score` with an input-only JSON snippet from the [README](../README.md). Explain that output values require a trained model.
2. **CPU execution path:** show the architecture graphic and explain the boundary: no answer-token decoding, but one native causal row per question and repeated shared-state work for multiple questions.
   Use the [neural model framework PNG](figures/model-framework.png) to explain the decoder layer, branch mask and pointer-head readout.
3. **Reproduction invitation:** link the [public-base JSON](benchmark-data/public-bitnet-base-2026-09-27.json) and [measurement notes](BENCHMARKS.md). Explain the original GGUF revision, model SHA-256, 8 threads, 128 input tokens, 32 output tokens, 5 runs, CPU-only binary, Vulkan hybrid binary, and global GPU-memory sampling.

## Launch checks

- Open the repository link while signed out; verify the README, quick start, diagram, license, and source-only tag.
- Inspect the final post text and image in X before posting. Link directly to the repository.
- Do not state that bit-jev weights are downloadable or that its accuracy, memory, or speed have been measured publicly. Label every chart as a Microsoft public-base measurement.
- After posting, record the post URL in the release log or launch notes and answer technical questions with the actual source paths and limitations.
