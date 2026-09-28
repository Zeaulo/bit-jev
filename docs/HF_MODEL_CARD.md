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

[中文模型卡](README.zh-CN.md) · [Source repository](https://github.com/Zeaulo/bit-jev) · [pip package](https://pypi.org/project/bit-jev/)

![bit-jev binary trail and accelerating rocket logo](bit-jev-logo.png)

On Windows x64 with Python 3.11/3.12, install the [official wheel](https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl) with the interpreter that will run inference, then check the distribution version and import path:

```bash
python -m pip install --upgrade --no-cache-dir "https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl"
python -c "from importlib.metadata import version; import bit_jev; print(version('bit-jev'), bit_jev.__file__)"
```

The version should be `0.11.10`, and the import path should point into the active environment's `site-packages/bit_jev`. Then run the Python example below, which does not depend on the `bit-jev-demo` console script being on PATH. The first model load downloads roughly 1.19 GB; later runs reuse the cache. See the [install guide](https://github.com/Zeaulo/bit-jev/blob/main/docs/GGUF_PACKAGE.md) for stale mirrors and mixed environments.

![bit-jev training, distillation, quantization, and CPU/Vulkan inference](project-flow.en.png)

The ternary symbols in the flow refer only to quantized BitLinear weights; training and inference are separate flows.

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

Install the Python package. On Windows x64 with an AVX2 CPU, the bit-jev 0.11.10 wheel includes precompiled CPU and Vulkan GPU runners. For `device="cpu"` or `device="gpu"`, inference needs no Git, CMake, compiler, or Vulkan SDK. Vulkan needs a compatible graphics driver that supplies `vulkan-1.dll`. The roughly 1.19 GB model still downloads on first use. Other platforms and CUDA build from pinned source and require Git, CMake 3.28+, and a C++17 compiler; CUDA needs the CUDA Toolkit.

```python
from bit_jev.gguf import BitJev

request = {
    "state": "A customer reports a duplicate charge on the same order.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle it?",
            "criteria": {"billing": "Payment and refunds", "shipping": "Delivery"},
        }
    },
}

with BitJev.from_pretrained(device="cpu", threads=8) as model:
    result = model.infer(request)
    print(result["answers"])
    print(result["latency_ms"])
```

`device="gpu"` selects Vulkan; `device="cuda"` selects a CUDA build. `infer()` reuses the resident model. See the [GGUF package guide](https://github.com/Zeaulo/bit-jev/blob/main/docs/GGUF_PACKAGE.md) for CLI, offline directories, and build details. This model repository contains model files; both precompiled runners are distributed in the PyPI Windows x64 wheel.

## AutoDL case study: EPYC 9654 CPU and RTX 5090 on another host

The measurements below use one fixed development request with 703 input tokens and 77 options. Native inference timing excludes model loading. The latency chart pairs a new EPYC 9654 container measurement with a 32-core quota and 32 native I2_S threads with a historical RTX 5090 FP16 mixed-precision PyTorch path from another host. The memory chart retains the original same-host Xeon Gold 6459C / RTX 5090 observations. These GPU numbers do not benchmark the pip package's GGUF/Vulkan or GGUF/CUDA path.

| Path | Mean inference time | Repetitions | Observed memory |
| --- | ---: | ---: | ---: |
| EPYC 9654, 32-core quota / 32 threads, I2_S | 1,954.46 ms | 6 | 1,625.88 MiB peak process RSS |
| Xeon Gold 6459C, 8 threads (historical) | 3,127.88 ms | 3 | 1,622.74 MiB peak process RSS |
| Xeon Gold 6459C, 16 threads (historical) | 1,972.17 ms | 3 | 1,624.52 MiB peak process RSS |
| RTX 5090 on another host, FP16 | 86.56 ms | 5 | 4,935.53 MiB peak GPU allocation |

![bit-jev AutoDL single-question inference latency comparison](bit-jev-case-speed.en.png)

![bit-jev AutoDL peak memory comparison](bit-jev-case-memory.en.png)

[中文图表](bit-jev-case-speed.zh-CN.png)

On the same EPYC container, a supplemental 16-thread run averaged 3,210.12 ms, so 32 threads were about 1.64 times faster. The CPU and GPU paths in the chart ran on different hosts and used different weight formats and numeric precision; their ratio is not an isolated hardware speedup. CPU RSS and GPU allocation are different measures. This small case study is not a general throughput or accuracy claim. The [EPYC 32-thread samples](https://github.com/Zeaulo/bit-jev/blob/main/docs/benchmark-data/bit-jev-epyc9654-cpu32-2026-09-28.json) and the original same-host case in `benchmark_case_autodl.json` include no input, option text, or predictions. Generated tokens/s does not apply because the model scores options and emits structured decisions. No auditable held-out report for accuracy, Brier, NLL, or ECE is available, so no such values are claimed.

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

