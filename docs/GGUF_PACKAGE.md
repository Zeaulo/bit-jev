# `bit-jev` pip package: GGUF loading and inference

[中文](GGUF_PACKAGE.zh-CN.md) · [Hugging Face model card](https://huggingface.co/jinghao1632/bit-jev-2b-distilled)

On Windows x64 with Python 3.11/3.12, install the official wheel directly and verify the distribution version and import path with the interpreter you will use for inference:

```bash
python -m pip install --upgrade --no-cache-dir "https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl"
python -c "from importlib.metadata import version; import bit_jev; print(version('bit-jev'), bit_jev.__file__)"
```

The second command should show `0.11.10` and a `site-packages/bit_jev/__init__.py` path in the active environment. The [complete Python quick start](../README.en.md#quick-start) can run immediately after installation; it does not require a console script on PATH. To run the bundled customer-routing question, use `python -m bit_jev.demo --threads 8` (`--device gpu` selects Vulkan). The first model load downloads roughly 1.19 GB; subsequent runs reuse the cache.

Install with `pip install bit-jev`. Version 0.11.10 provides precompiled CPU and Vulkan GPU runners in the Windows x64 wheel for AVX2 processors. Inference with `device="cpu"` or `device="gpu"` does not need Git, CMake, a C++ compiler, or a Vulkan SDK. Vulkan GPU inference does require a working graphics driver that supplies the `vulkan-1.dll` loader. The 1.19 GB GGUF still downloads on first use. Other platforms and `device="cuda"` build the runner from pinned source and need [Git](https://git-scm.com/install/), [CMake 3.28+](https://cmake.org/download/), and a C++17 compiler ([Windows C++ Build Tools](https://learn.microsoft.com/cpp/build/vscpp-step-0-installation)); an explicit Vulkan source build also needs Vulkan headers, libraries, and `glslc`, while CUDA builds need the CUDA Toolkit. Git fetches and verifies fixed BitNet and llama.cpp commits and applies the ReLU² patch only for source builds. Passing `binary=` selects your own runner, while `native_source=` forces a build from prepared local source. Both bundled runners disable host-specific CPU tuning and require AVX2; the Vulkan runner links the MinGW runtime statically and loads the system Vulkan loader.

If your pip configuration uses a mirror that still reports an older version, run the explicit official-index command above. Check both `importlib.metadata.version('bit-jev')` and `bit_jev.__file__`: `bit_jev.__version__` alone can come from a different source checkout and does not prove that the active interpreter installed the corresponding wheel. If pip downgrades `peft` to 0.14.0 and reports an Unsloth conflict, it has selected an old bit-jev release; version 0.11.10 has no `peft` dependency for default inference. Use a fresh Python 3.11/3.12 environment rather than altering an existing Unsloth environment. Windows GPU users should update their graphics driver if the Vulkan loader is missing. SDK headers and `glslc` are needed only when explicitly building a Vulkan runner from source; see the repository's [GGUF build guide](https://github.com/Zeaulo/bit-jev/blob/main/docs/GGUF_PACKAGE.md).

If the official simple index also serves a stale cache just after release, Windows x64 users can install the [official PyPI wheel](https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl) directly: `python -m pip install --upgrade "https://files.pythonhosted.org/packages/b1/2c/d044c5bccdf4d952e09e1e7a483145cb4fd4cc311da6ea01ebc1bb871b1c/bit_jev-0.11.10-py3-none-win_amd64.whl"`. SHA-256: `ffc8ad7756ad9b221536ef5dcddfebead025aa66abbf06244a15e3983169b1b4`.


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
