# bit-jev

**Install, run a decision, then build your own structured requests.** bit-jev scores explicit options over a BitNet backbone and returns answers and probabilities without generating answer text token by token.

![bit-jev ternary inputs and decision engine](https://raw.githubusercontent.com/Zeaulo/bit-jev/main/docs/figures/decision-engine.png)

[GitHub documentation](https://github.com/Zeaulo/bit-jev) · [Hugging Face model](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) · [ModelScope model](https://www.modelscope.cn/models/JingHao9616/bit-jev-2b-distilled) · [中文说明](https://github.com/Zeaulo/bit-jev/blob/main/README.md)

## Try it

```bash
python -m pip install --upgrade bit-jev
bit-jev-demo
```

The built-in demo prints a real model answer and latency. The first run downloads the 1.19 GB GGUF from Hugging Face or, if that connection fails, ModelScope; later runs reuse the cache. Installation itself does not download model weights. Use `bit-jev-demo --source modelscope` to select ModelScope directly, or `--device gpu` for Vulkan.

The Windows x64 wheel includes precompiled CPU and Vulkan GPU runners for AVX2 processors. Inference on those machines needs no Git, CMake, C++ compiler, or Vulkan SDK. Vulkan still needs a compatible graphics driver and its `vulkan-1.dll` runtime. Other platforms build the native runner on demand and require [Git](https://git-scm.com/install/), [CMake 3.28+](https://cmake.org/download/), and a C++17 compiler ([Windows C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation)). Source builds of Vulkan also need its SDK; CUDA builds need a CUDA Toolkit. Both precompiled programs use pinned BitNet and llama.cpp source with the ReLU² runtime patch and carry their MIT license notices.

## Resident inference

```python
from bit_jev.gguf import BitJev

request = {
    "state": "A customer reports a duplicate charge.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payment and refund issues", "shipping": "Delivery issues"},
        }
    },
}

with BitJev.from_pretrained(device="cpu", threads=8) as model:
    result = model.infer(request)
    print(result["answers"], result["latency_ms"])
```

Use `device="gpu"` for Vulkan or `device="cuda"` for an NVIDIA CUDA build. GPU requests fail clearly if a backend or visible GPU is unavailable. Pass `source="modelscope"` to select ModelScope directly, or pass a local model directory for offline inference. `binary="/path/to/bit-jev-cpu"` selects an existing native runner. The model stays loaded for repeated `infer()` calls; `latency_ms` reports native compute only, excluding download, build, loading, encoding, and IPC.

The CLI accepts UTF-8 JSONL input:

```bash
bit-jev --device cpu --input requests.jsonl --output results.jsonl
```

Training and distillation dependencies are optional: `pip install 'bit-jev[train]'`. The native runner evaluates one causal row per question; multiple questions repeat the shared state. The repository's Apache-2.0 license covers source code, while the checkpoint has no standalone open-weights license. The model card documents Yelp training-data provenance and its unresolved permission request.
