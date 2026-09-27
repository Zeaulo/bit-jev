# bit-jev: architecture and model-artifact status

## Release status

The current release contains source, public-base measurements, and a documented bit-jev case study. The Yelp-trained checkpoint, I2_S GGUF, pointer-head weights, and checkpoint-derived benchmark bundle remain pending the written data-rights response requested from Yelp. The separate [public BitNet backbone measurements](BENCHMARKS.md#microsoft-public-bitnet-base-independent-measurement) do not measure bit-jev classification.

This page documents the **code interface and requirements for a future model artifact**. It is not a performance card for a model available to download.

## Architecture

The implementation is inspired by [Kev](https://github.com/jaredpalmer/kev)'s structured decision interface and uses the [Microsoft BitNet b1.58 2B family](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16) as the backbone design. A request provides shared `state` and typed questions:

- `choice`: score caller-supplied candidate options.
- `noul`: score a yes/no decision.
- `score`: score ordered categories.

The pointer head reads the option-boundary and decision hidden states and produces option logits and probabilities. It does not produce answer text. The PyTorch packed implementation uses a block-causal attention mask to isolate question branches while sharing state computation. The native CPU implementation evaluates **one causal row per question**, so a multiquestion request repeats the shared-state work.

The [English highlights figure](figures/model-highlights.en.svg) follows the code's training-to-deployment order: `train.py` trains LoRA and the pointer head on the BitNet BF16 base; `distill.py export_teacher` records option logits from the trained model; `distill.py train` trains a full-backbone student without LoRA and uses the fine-tuned pointer head; `export_distilled.py` exports a compatible student backbone to I2_S GGUF. Ternary describes **quantized BitLinear weights**, not every tensor. The CPU numbers in the figure measure the separate Microsoft public base model, not this trained pipeline.

The native runner reads JSONL and returns JSONL. It is not an HTTP service. The PyTorch service in `core/bit_jev/serve.py` is a separate execution path.

## What an authorized CPU artifact must supply

A future CPU package, or a user-provided compatible model, must supply:

| File or component | Purpose |
| --- | --- |
| Matching tokenizer and configuration | Encode the request with the expected reserved markers and model dimensions |
| I2_S GGUF backbone | Run the quantized BitNet model through the native CPU binary |
| Float32 pointer-head sidecar | Score the option and decision hidden states |
| Pointer metadata | Record delimiter IDs, head layout, and temperature |
| Provenance and hashes | Identify the base, training data and rights, export procedure, and exact files |

The [CPU quick start](CPU_QUICKSTART.md) builds the code and shows the command shape for artifacts the user is authorized to use. It does not provide a ready-to-run model package.

## Intended use and limits

The code supports research into routing, yes/no decisions, and ordered ratings with explicit options. It is not a general chat generator. The speed or memory of a resulting system depends on the particular weights, build, CPU, question count, input length, and candidate count. The present source release provides **no checkpoint accuracy, calibration, latency, or resident-memory guarantee**.

Before deploying any trained artifact, evaluate it on a permitted dataset that matches the application. Check option ordering, numerical parity between CPU and reference paths, probability calibration, and error costs. Human review is needed for consequential decisions until the system has been validated in that setting.

## Requirements for a future public model card

A public artifact should identify the immutable base and teacher revisions, every training/evaluation source and its usage rights, training configuration, file hashes, supported input contract, exact inference precision, hardware measurements, per-split quality, calibration, known failure modes, and license terms. The [benchmark protocol](BENCHMARKS.md) states how to report timing and memory without mixing load, warmup, and resident inference.

Repository code is under [Apache-2.0](../LICENSE). Upstream source and base-model notices appear in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Those licenses do not by themselves settle rights to distribute a derivative trained on separately licensed data.
