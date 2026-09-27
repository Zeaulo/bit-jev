# bit-jev core

This directory contains the implementation of a structured decision model: a
BitNet b1.58 backbone, Kev-style block-causal request encoding, and a pointer
head that scores candidate answers. **This GitHub repository contains source,
not trained bit-jev weights.** The I2_S CPU package and its model card are
published at <https://huggingface.co/jinghao1632/bit-jev-2b-distilled>.

The model emits a probability distribution over supplied options. It does not
decode an answer token by token. The packed PyTorch implementation can process
multiple question branches in one forward pass. The current native CPU runner
evaluates one causal row per question and recomputes the shared state for each
row. Its runtime therefore depends on the number and length of questions and
options, as well as the CPU and thread configuration.

## Source layout

| Path | Responsibility |
| --- | --- |
| `bit_jev/api.py` | Validate TypeSafe requests and format Choice, Noul, and Score answers. |
| `bit_jev/model.py` | Encode delimiter tokens, build the branch mask, run the backbone, and score options with the pointer head. |
| `bit_jev/train.py` | Fine-tune the backbone and pointer head using labelled decision requests. |
| `bit_jev/eval.py` | Evaluate a locally supplied checkpoint and dataset. |
| `bit_jev/checkpoint.py` | Restore local run directories and their head metadata. |
| `bit_jev/export_distilled.py` | Convert a compatible local checkpoint to an I2_S GGUF backbone and pointer metadata. |
| `bit_jev/cpu_head.py` | Write the pointer head as a native float32 sidecar. |
| `bit_jev/cpu.py` | Encode isolated CPU rows, call the persistent native process, and format its answers. |
| `bit_jev/serve.py` | Expose the PyTorch inference path through a local FastAPI endpoint. |
| `native/main.cpp` | Load a local GGUF and head, evaluate causal rows, and return JSONL results. |
| `native/CMakeLists.txt` | Build the CPU executable against the pinned bitnet.cpp source. |
| `data/api_request.json` | Example request shape; it is not a model output. |

## Decision path

A request is serialized into a state segment followed by question branches.
BitNet's reserved delimiter tokens mark the state, each question, each option,
and the decision position. In the packed PyTorch path, a block-causal
attention mask lets a branch read the shared state and its own tokens while
preventing one question from reading another. The pointer head compares each
option-close hidden state with that question's decision hidden state, then
applies softmax. This is classification over explicit candidates, not text
generation.

The native CPU path uses a separate causal row for every question because the
current bitnet.cpp runner does not implement the packed branch mask. Each row
contains the state and one question. The native process reads the option-close
and decision hidden states and applies the same pointer-head schema. Repeating
the state for every row is a known performance cost and an area for future work.

## Download the model

Read the Hugging Face model card before downloading. It documents the Yelp
training-data provenance, unanswered permission request, and the absence of a
standalone open-weights license. The model package does not contain this
directory's native runner binary.

## Prepare and inspect source

From the repository root, install the Python package and obtain the pinned
BitNet and llama.cpp revisions:

```powershell
python -m pip install -e ./core
python test/bootstrap_bitnet.py
python test/bootstrap_bitnet.py --check
```

`test/bootstrap_bitnet.py` applies the checked-in converter and ReLU² patches
to local upstream source. It verifies the expected revisions and patch state.
The upstream repositories are kept under the ignored `learning/` directory.
The bootstrap command may require network access. A previous local-mirror
bootstrap and native build passed; a direct public-network submodule download
has not been verified on every environment.

With CMake 3.28+ and a compatible C++17 compiler, the native executable can
be built without any bit-jev model checkpoint:

```powershell
cmake -S core/native -B core/build/bit-jev-cpu -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build core/build/bit-jev-cpu --target bit-jev-cpu -j 4
python -m bit_jev.cpu --help
```

The native program cannot classify requests until a compatible, separately
obtained checkpoint has been trained and exported. The source release does
not provide such a checkpoint.

## Local checkpoint and export contract

The historical training and export code remains available for experiments
with data and model artifacts that you are permitted to use. Review the
licenses and terms of every input dataset and teacher model before training
or distributing any resulting weights or metrics. The previous local run used
data with unresolved redistribution terms; its weights and derived reports
are intentionally absent from this public release.

A compatible local distilled checkpoint stores the full backbone state,
pointer-head weights, tokenizer, and model configuration. The export command
accepts paths supplied by the operator:

```powershell
python -m bit_jev.export_distilled --run <local-run-directory> --out <local-export-directory>
```

An export produces `backbone-i2_s.gguf`, `head.pt`, `head.f32`,
`pointer.json`, and `manifest.json`. The GGUF holds the backbone;
`head.f32` holds the native pointer weights; `pointer.json` records
delimiter IDs, the head dimension, and temperature. These are generated
artifacts, not files in this repository.

For an operator-owned compatible export, the CPU command reads UTF-8 JSONL
requests and writes TypeSafe answers:

```powershell
python -m bit_jev.cpu --run <local-run-directory> --artifact <local-export-directory> --binary <local-native-binary> --input <local-input.jsonl> --out <local-output.jsonl> --threads 4 --batch 128
```

The placeholder paths above must be replaced with real local artifacts. The
command does not download a model. The same restriction applies to
`bit_jev.serve` and `bit_jev.eval`.

## Licensing and attribution

The bit-jev source is licensed under Apache-2.0. The BitNet base model,
bitnet.cpp source, Kev architecture, datasets, and any teacher model retain
their own licenses and usage terms. See the repository
`THIRD_PARTY_NOTICES.md` for attribution. This source license does not grant
rights to distribute third-party datasets or any checkpoint trained on them.
