# bit-jev: architecture and model-artifact status

## Release status

The bit-jev-2b-distilled I2_S package and sanitized AutoDL single-question measurements are published on [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled). The training set includes Yelp review records. A permission request was sent; as of 2026-09-28, no written reply had arrived. The model card discloses this status and specifies no standalone open-weights license. Separate [Microsoft public-base measurements](BENCHMARKS.md#microsoft-public-bitnet-base-independent-measurement) test the original backbone, not bit-jev classification.

This page documents the code interface. See the [Hugging Face model card](HF_MODEL_CARD.md) for training, the published package, the single-question measurement case, and its limitations.

## Architecture

The implementation is inspired by [Kev](https://github.com/jaredpalmer/kev)'s structured decision interface and uses the [Microsoft BitNet b1.58 2B family](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16) as the backbone design. A request provides shared `state` and typed questions:

- `choice`: score caller-supplied candidate options.
- `noul`: score a yes/no decision.
- `score`: score ordered categories.

The pointer head reads the option-boundary and decision hidden states and produces option logits and probabilities. It does not produce answer text. The PyTorch packed implementation uses a block-causal attention mask to isolate question branches while sharing state computation. The native CPU implementation evaluates **one causal row per question**, so a multiquestion request repeats the shared-state work.

The [English highlights figure](figures/model-highlights.en.svg) follows the code's training-to-deployment order: `train.py` trains LoRA and the pointer head on the BitNet BF16 base; `distill.py export_teacher` records option logits from the trained model; `distill.py train` trains a full-backbone student without LoRA and uses the fine-tuned pointer head; `export_distilled.py` exports a compatible student backbone to I2_S GGUF. Ternary describes **quantized BitLinear weights**, not every tensor. The CPU numbers in the figure measure the separate Microsoft public base model, not this trained pipeline.

The native runner reads JSONL and returns JSONL. It is not an HTTP service. The PyTorch service in `core/bit_jev/serve.py` is a separate execution path.

## Published CPU package

A future CPU package, or a user-provided compatible model, must supply:

| File or component | Purpose |
| --- | --- |
| Matching tokenizer and configuration | Encode the request with the expected reserved markers and model dimensions |
| I2_S GGUF backbone | Run the quantized BitNet model through the native CPU binary |
| Float32 pointer-head sidecar | Score the option and decision hidden states |
| Pointer metadata | Record delimiter IDs, head layout, and temperature |
| Provenance and hashes | The Hub card records the base, training data status, export procedure, and per-file SHA-256 |

The [CPU quick start](CPU_QUICKSTART.md) shows how to build the runner, download the Hub files, and invoke the model.

## Intended use and limits

The code supports research into routing, yes/no decisions, and ordered ratings with explicit options. It is not a general chat generator. The published AutoDL results come from one development request and do not establish accuracy or general speed. Actual speed and memory depend on weights, build, CPU, question count, input length, and candidate count.

Before deploying any trained artifact, evaluate it on a permitted dataset that matches the application. Check option ordering, numerical parity between CPU and reference paths, probability calibration, and error costs. Human review is needed for consequential decisions until the system has been validated in that setting.

## Published evaluation and license status

The public measurements record hardware, request shape, repetitions, quantization/precision, and memory definitions. No auditable held-out Accuracy, Brier, NLL, or ECE report is currently available. The [benchmark protocol](BENCHMARKS.md) links to sanitized timing samples and explains load, warmup, resident inference, and memory definitions.

Repository code is under [Apache-2.0](../LICENSE). Upstream source and base-model notices appear in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The code license does not automatically apply to the Hugging Face weights or settle distribution rights for a checkpoint trained on separately licensed Yelp data.
