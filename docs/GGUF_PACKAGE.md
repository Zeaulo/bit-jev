# `bit-jev` pip package: GGUF loading and inference

[中文](GGUF_PACKAGE.zh-CN.md) · [Hugging Face model card](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

Install with `pip install bit-jev`. Version 0.9.9 provides a precompiled CPU runner in the Windows x64 wheel for AVX2 processors, so CPU inference there does not need Git, CMake, or a C++ compiler. The 1.19 GB GGUF still downloads on first use. Other platforms and GPU backends build the runner from pinned source and need [Git](https://git-scm.com/install/), [CMake 3.28+](https://cmake.org/download/), and a C++17 compiler ([Windows C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation)); Vulkan requires a Vulkan SDK and CUDA requires a CUDA Toolkit. Git fetches and verifies fixed BitNet and llama.cpp commits and applies the ReLU² patch only for source builds. Passing `binary=` selects your own runner, while `native_source=` forces a build from prepared local source. The wheel's runner was built without host-specific CPU tuning and with the MSVC runtime linked statically; AVX2 remains required.

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

`from_pretrained()` waits for model loading. `infer()` reuses the resident model and returns answers, logits, probabilities, native inference latency in milliseconds, and the selected device. Native latency excludes download, compilation, model loading, tokenizer encoding, and IPC. `gpu` selects Vulkan; `cuda` selects an NVIDIA CUDA build. Use `gpu_index=1` on a multi-GPU host to select a visible device in the native subprocess. GPU requests fail if no GPU backend or visible GPU is available instead of silently falling back to CPU. Vulkan may keep some I2_S weights in system RAM.

An existing local model directory can be passed as the first argument. Pass `binary="/path/to/bit-jev-cpu"` to use a previously built runner. The default native build cache is `~/.cache/bit-jev`; use `BIT_JEV_CACHE` or `native_cache` to change it. The CLI accepts JSONL requests:

```bash
bit-jev --model jinghao1632/bit-jev-2b-distilled --device cpu --input requests.jsonl --output results.jsonl
```

For training dependencies, install `bit-jev[train]`. The repository's Apache-2.0 license covers code only. The checkpoint includes training derived from Yelp reviews and has no standalone open-weights license; read its model card before use. The historical AutoDL RTX 5090 timing is for a separate FP16 PyTorch path and is not a GGUF GPU benchmark.
