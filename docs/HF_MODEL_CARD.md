---
base_model:
  - microsoft/bitnet-b1.58-2B-4T-bf16
library_name: bit-jev
tags:
  - bitnet
  - structured-decision
  - pointer-head
  - cpu-inference
  - gpu-inference
  - knowledge-distillation
  - yelp
---

# bit-jev-2b-distilled

[中文模型卡](HF_MODEL_CARD.zh-CN.md) · [Source repository](https://github.com/Zeaulo/bit-jev) · [pip package](https://pypi.org/project/bit-jev/)

![bit-jev training, distillation, and I2_S CPU inference flow](figures/project-cover.en.png)

The ternary symbols represent quantized BitLinear weights; training and inference are separate flows.

> This repository provides the I2_S GGUF model package for bit-jev, usable with the `bit-jev` Python package on CPU or GPU. The checkpoint was trained on a multi-source decision dataset that includes Yelp review records. A request for permission covering derivative weights and metrics has been sent to Yelp; as of 2026-09-28, no written reply has been received. The Apache-2.0 license for the source repository does not automatically apply to this checkpoint.

## Model summary

bit-jev-2b-distilled is a structured-decision student model. An input contains a shared `state` and one or more questions. A pointer head scores explicit options and returns a choice, probabilities, or an ordered rating. It does not generate a natural-language answer token by token.

| Type | Input | Structured output |
| --- | --- | --- |
| `choice` | Named options and descriptions | Selected option, scores, probabilities |
| `noul` | Yes/no question | Scores and probabilities for both labels |
| `score` | Ordered ratings and descriptions | Rating scores, expected value, probabilities |

## Training and export

```text
Microsoft BitNet b1.58 2B BF16
        ↓ LoRA fine-tuning + pointer-head training
Initial bit-jev model and pointer head
        + candidate logits from the Kev 9B teacher
Full-student distillation + pointer head
        ↓ I2_S quantized export
I2_S GGUF + float32 pointer head
```

This release contains the I2_S artifacts for the native CPU runner, matching tokenizer/configuration, and pointer metadata. The recorded training configuration uses 3,144 steps, 2 epochs, BF16, learning rate `2e-5`, batch size 2, gradient accumulation 4, distillation temperature 2.0, and distillation weight 1.0. The inference-head temperature is 2.35 in `pointer.json`.

## Files and size

| File | Purpose | Size |
| --- | --- | ---: |
| `backbone-i2_s.gguf` | Quantized BitNet backbone | 1,187,288,192 bytes (about 1.106 GiB) |
| `head.f32` | Float32 pointer head | 5,244,948 bytes |
| `pointer.json` | Head shape, boundary tokens, and temperature | 770 bytes |
| Tokenizer/configuration files | Request encoding and model configuration | about 17.2 MB |

`SHA256SUMS.json` lists sizes and SHA-256 hashes for inference-package files; it does not hash itself. BF16 safetensors shards, training data, training logs, and raw reviews are not included.

## Inference

Install the Python package. On first load, it downloads the GGUF, pointer head, and tokenizer, then builds the pinned native runner. Git, CMake 3.28+, and a C++17 compiler are required; Vulkan GPU mode needs a Vulkan SDK and CUDA mode needs a CUDA Toolkit.

```bash
pip install bit-jev
```

```python
from bit_jev import BitJev

request = {"state": "A customer reports a duplicate charge.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": {"billing": "Payment and refunds", "shipping": "Delivery"}}}}
with BitJev.from_pretrained(device="cpu", threads=8) as model:
    print(model.infer(request)["answers"])
```

`device="gpu"` selects Vulkan; `device="cuda"` selects a CUDA build. `infer()` reuses the resident model. See the [GGUF package guide](https://github.com/Zeaulo/bit-jev/blob/main/docs/GGUF_PACKAGE.md) for CLI, offline directories, and build details. This model repository contains no prebuilt runner binary.

## AutoDL case study: Xeon Gold 6459C / RTX 5090

The measurements below use one fixed development request with 703 input tokens and 77 options. Inference timing excludes model loading. CPU uses the native I2_S path; GPU uses an experimental FP16 mixed-precision PyTorch path. These GPU numbers do not benchmark the new pip package's GGUF/Vulkan or GGUF/CUDA path.

| Path | Mean inference time | Repetitions | Observed memory |
| --- | ---: | ---: | ---: |
| CPU, 8 threads | 3,127.88 ms | 3 | 1,622.74 MiB peak process RSS |
| CPU, 16 threads | 1,972.17 ms | 3 | 1,624.52 MiB peak process RSS |
| RTX 5090 | 86.56 ms | 5 | 4,935.53 MiB peak GPU allocation |

![bit-jev AutoDL single-question latency and memory](figures/bit-jev-autodl-case.en.svg)

[中文图表](figures/bit-jev-autodl-case.zh-CN.svg)

For this one request, the 16-thread CPU to RTX 5090 latency ratio is about 22.8. The paths use different weight formats and numeric precision, so this is not an isolated hardware speedup. CPU RSS and GPU allocation are different measures. This small case study is not a general throughput or accuracy claim. Sanitized timings are in `benchmark_case_autodl.json`; the input, options, and predictions are not included. Generated tokens/s does not apply because the model scores options and emits structured decisions. No auditable held-out report for accuracy, Brier, NLL, or ECE is available, so no such values are claimed.

## Data provenance and use

Training used the local multi-source `decision-v7` decision set, which includes Yelp review records. The repository does not contain raw Yelp records. A request for permission to publish derivative weights and metrics has been sent to Yelp; as of 2026-09-28, no written reply has been received. Downloaders should review the applicable Yelp dataset terms and their effect on derivative weights.

**No standalone open-weights license is granted for this model.** The source repository's Apache-2.0 license, Microsoft's base-model license, and Kev's source license apply to their respective works; they do not grant rights to Yelp data or this derivative checkpoint. This card documents the project's current release status and is not legal advice.

## Evaluation scope and limitations

- The only public bit-jev results here are the single-request latency and memory case above. The independent Microsoft BitNet base-model benchmark measures a different model and is not a score for this checkpoint.
- No Accuracy, Brier, NLL, ECE, or confidence interval is available across complete train/test source splits.
- The request is a development case with few repetitions. Results do not generalize to long inputs, multi-question requests, or other hardware.
- The native CPU runner executes one causal row per question and recomputes shared state for multiple questions.
- Choices may depend on option wording, order, length, and training distribution. Validate on target data and retain human review for consequential decisions.

## Citation

```bibtex
@misc{bitjev2026,
  title = {bit-jev-2b-distilled},
  author = {Zeaulo},
  year = {2026},
  url = {https://huggingface.co/jinghao1632/bit-jev-2b-distilled}
}
```

