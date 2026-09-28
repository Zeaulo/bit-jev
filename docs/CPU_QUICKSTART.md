# CPU source quick start

This guide covers a manual native I2_S CPU source build and JSONL inference. For a pip install with automatic model download and GPU options, see the [GGUF package guide](GGUF_PACKAGE.md). The bit-jev-2b-distilled checkpoint is on [Hugging Face](https://huggingface.co/jinghao1632/bit-jev-2b-distilled), outside the GitHub repository. The separate [Microsoft public-base benchmark](BENCHMARKS.md#microsoft-public-bitnet-base-independent-measurement) measures a different backbone. Read the model card for Yelp data provenance, measurement limits, and license status before downloading.

The Python launcher encodes requests and manages a persistent native process. The native binary performs backbone inference and pointer-head scoring. It does not run an HTTP server.

## 1. Clone and install

Use Python 3.11 or 3.12, Git, CMake 3.28 or newer, and a C++17 compiler. Install a PyTorch build appropriate for your platform using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the package. From PowerShell:

~~~powershell
git clone https://github.com/Zeaulo/bit-jev.git
Set-Location bit-jev
python -m pip install -e './core'
~~~

The native bootstrap needs network access to fetch pinned upstream source on its first run. It applies the repository's converter and ReLU² compatibility patches, then verifies the resulting source. Run it **before** CMake:

~~~powershell
python test/bootstrap_bitnet.py
python test/bootstrap_bitnet.py --check
~~~

On Windows with MinGW Makefiles:

~~~powershell
cmake -S core/native -B core/build/bit-jev-cpu -G 'MinGW Makefiles' -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build core/build/bit-jev-cpu --target bit-jev-cpu -j 4
~~~

On Linux with the default generator:

~~~bash
python3 test/bootstrap_bitnet.py
python3 test/bootstrap_bitnet.py --check
cmake -S core/native -B core/build/bit-jev-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build core/build/bit-jev-cpu --target bit-jev-cpu -j 4
~~~

`GGML_NATIVE=ON` tunes the binary for the build machine's CPU; rebuild on a different target CPU. The Windows binary is `core/build/bit-jev-cpu/bit-jev-cpu.exe`; on Linux it has no `.exe` suffix. A full first-time upstream download may be affected by GitHub network availability.

## 2. Prepare an input

Save one complete UTF-8 JSON object per line in `test/my-request.jsonl`. This example is **input only**:

~~~json
{"state":"A customer reports a duplicate charge.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payment and refund issues","shipping":"Delivery issues"}}}}
~~~

The `questions` object must be nonempty. The request can contain `choice`, `noul`, and `score` questions; [the architecture page](MODEL_CARD.md) describes the readout. No example output or expected prediction is supplied because the public source release has no trained bit-jev model.

## 3. Download and run the published I2_S model

Download the model files from Hugging Face. `--run` points to the tokenizer/config directory, while `--artifact` points to the directory containing `backbone-i2_s.gguf` and `head.f32`. This package keeps both in one directory, so both flags use the same path.

~~~powershell
hf download jinghao1632/bit-jev-2b-distilled --local-dir './models/bit-jev-2b-distilled'
python -m bit_jev.cpu `
  --run './models/bit-jev-2b-distilled' `
  --artifact './models/bit-jev-2b-distilled' `
  --binary './core/build/bit-jev-cpu/bit-jev-cpu.exe' `
  --input './test/my-request.jsonl' `
  --out './test/my-result.jsonl' `
  --threads 8 --batch 128
~~~

On Linux, use the binary path without `.exe` and normal shell continuation syntax. The result includes answers, logits, probabilities, and native compute latency. The Yelp data permission request for this checkpoint is still awaiting a written response, and no open-weights license is specified in its model card; the code's Apache-2.0 license does not cover the weights. Keep generated requests and results under `test/` in this workspace.

The native runner evaluates **one causal row per question**. It does not generate answer tokens, but multiquestion requests repeat shared-state computation and long option lists increase input work. Use the [benchmark protocol](BENCHMARKS.md) to measure model load, resident latency, and process memory separately.
