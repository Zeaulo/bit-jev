# CPU source quick start

This guide builds the native I2_S runner and shows the JSONL inference interface. **v0.6.7 remains a source-only bit-jev release**: there is no public bit-jev weight archive or saved answer to reproduce. The separate [Microsoft public-base benchmark](BENCHMARKS.md#microsoft-public-bitnet-base-independent-measurement) measures backbone execution, not classification. Running inference requires your own compatible, authorized tokenizer, I2_S GGUF backbone, and trained pointer-head sidecar. The Microsoft BitNet base model by itself does not provide bit-jev's trained decision head.

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

## 3. Run with your authorized artifacts

After you have prepared a compatible trained model, replace both artifact placeholders below with its **local directories**. `--run` points to the matching tokenizer directory. `--artifact` points to the directory containing `backbone-i2_s.gguf` and `head.f32`; pointer metadata should accompany an artifact to document its compatibility, although the CLI reads the GGUF and head files directly. The directories may be the same if your own package combines them.

~~~powershell
python -m bit_jev.cpu `
  --run './path/to/authorized-run' `
  --artifact './path/to/authorized-export' `
  --binary './core/build/bit-jev-cpu/bit-jev-cpu.exe' `
  --input './test/my-request.jsonl' `
  --out './test/my-result.jsonl' `
  --threads 8 --batch 128
~~~

On Linux, use the binary path without `.exe` and normal shell continuation syntax. The result includes answers, logits, probabilities, and native compute latency; the values depend on the supplied model. Keep generated requests and results under `test/` in this workspace.

The native runner evaluates **one causal row per question**. It does not generate answer tokens, but multiquestion requests repeat shared-state computation and long option lists increase input work. Use the [benchmark protocol](BENCHMARKS.md) to measure model load, resident latency, and process memory separately.
