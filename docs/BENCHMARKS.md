# Benchmark protocol and published case studies

This page separates the Microsoft public-base benchmark from the bit-jev AutoDL case study. The latter uses the local Yelp-trained checkpoint and remains a local review artifact until the requested data-rights response is received. The scripts in [test/](../test/) are measurement tools; their presence does not make a result reproducible without an authorized compatible model and input set.

bit-jev scores explicit options after reading the input. It does not decode answer text token by token, so generated-output tokens per second is not an appropriate throughput measure. Report **request latency** and, if useful, **input tokens per second** with the token-count definition stated.

## Microsoft public BitNet base independent measurement

The charts below measure only the [original public Microsoft BitNet b1.58 2B I2_S GGUF](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/tree/9f43072f69492cbd5bc5d5ebb085fec519686a93), revision `9f43072f69492cbd5bc5d5ebb085fec519686a93`. Both paths use the same 1,844,472,032-byte file, SHA-256 `13939ce5030319a35db346e5dba7a3a3bd599dfc18b113a2a97446ff964714c5`. The model file is not shipped here. This fixed earlier revision avoids mixing results with [reported anomalies in a newer official revision](https://github.com/microsoft/BitNet/issues/608).

![Public base prefill and decode throughput](figures/public-base-speed.svg)

![Public base RAM and VRAM observations](figures/public-base-memory.svg)

| Measurement | Ryzen 7 4800H, 8 threads, CPU-only binary | RTX 2060, Vulkan1 hybrid |
| --- | ---: | ---: |
| 128 input token prefill median | 2,276.20 ms; 56.23 tokens/s | 2,831.68 ms; 45.20 tokens/s |
| 32 output token decode median | 7,008.26 ms; 4.57 tokens/s | 9,000.01 ms; 3.56 tokens/s |
| Peak process RSS, including load and tests | 1,230.19 MiB | 1,955.56 MiB |
| Peak global GPU VRAM rise over pre-run baseline | 0 MiB | 722 MiB |
| Backend model-buffer log | CPU_Mapped 1,751.06 MiB | CPU_Mapped 1,124.80 MiB; Vulkan1 627.93 MiB |

The command uses `llama-bench -p 128 -n 32 -r 5 -t 8 -b 128 -ub 128`, with a warmup per phase and sequential CPU/GPU runs. Prefill and decode are separate **synthetic backbone workloads**; their times are not the end-to-end time for one question. Throughput excludes model loading. Process RSS is sampled every 50 ms through loading and testing; `nvidia-smi` global VRAM use is sampled every 250 ms. Windows WDDM did not expose reliable per-process Vulkan VRAM, so the VRAM figure is a global delta that may include other processes. All 31 layers were configured for offload, while I2_S work remained partly CPU mapped. This is neither pure GPU inference nor bit-jev classification. CPU was faster on this particular host and workload.

The [public JSON](benchmark-data/public-bitnet-base-2026-09-27.json) contains the five samples per phase, hardware, model and binary hashes. The [runner](../test/benchmark_public_base.py) writes a local raw report under `test/`; the [renderer](../test/render_public_base_figures.py) strips local paths and creates the public JSON and charts. For reproduction, run [`test/bootstrap_bitnet.py`](../test/bootstrap_bitnet.py) to fetch pinned BitNet/llama.cpp and apply the ReLU² patch, then apply the [original GGUF compatibility patch](../test/patches/public-bitnet25-compat.patch) inside `learning/bitnet/3rdparty/llama.cpp`. The latter handles the original GGUF's `bitnet-25` architecture name and separate `output.weight`. Build separate `llama-bench` binaries with `GGML_VULKAN=OFF` and `GGML_VULKAN=ON`, then run:

```powershell
python test/bootstrap_bitnet.py
git -C learning/bitnet/3rdparty/llama.cpp apply "$(Resolve-Path test/patches/public-bitnet25-compat.patch)"
cmake -S learning/bitnet/3rdparty/llama.cpp -B test/build-public-base -G "MinGW Makefiles" -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
cmake --build test/build-public-base --target llama-bench -j 8
cmake -S learning/bitnet/3rdparty/llama.cpp -B test/build-public-vulkan -G "MinGW Makefiles" -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
cmake --build test/build-public-vulkan --target llama-bench -j 8
python test/benchmark_public_base.py --model test/ggml-model-i2_s-original.gguf --cpu-binary test/build-public-base/bin/llama-bench.exe --gpu-binary test/build-public-vulkan/bin/llama-bench.exe --gpu-device Vulkan1 --threads 8 --prompt-tokens 128 --decode-tokens 32 --repetitions 5 --output test/public_base_llama_bench_final.json
python test/render_public_base_figures.py
```

The build example uses Windows MSYS2 UCRT; the Vulkan build also needs Vulkan headers and import libraries discoverable by CMake. Download the original GGUF from the fixed revision above into `test/`, and select the actual GPU device reported by `llama-bench --list-devices`. These measurements do not establish bit-jev request latency, accuracy, or a speed ratio against Kev.

## bit-jev AutoDL single-question case

![bit-jev CPU/GPU latency and memory for one question](figures/bit-jev-autodl-case.en.svg)

This measurement uses the bit-jev-2b-distilled checkpoint published on Hugging Face. One fixed development request encoded to 703 input tokens and 77 options. The native CPU I2_S path ran three times; the GPU FP16 mixed-precision path ran five times. Model loading is excluded from inference time; GPU warmup is also excluded.

| Path | Mean inference time | Repetitions | Peak memory observation |
| --- | ---: | ---: | ---: |
| CPU, 8 threads, native I2_S | 3,127.88 ms | 3 | 1,622.74 MiB peak process RSS |
| CPU, 16 threads, native I2_S | 1,972.17 ms | 3 | 1,624.52 MiB peak process RSS |
| RTX 5090, FP16 mixed precision | 86.56 ms | 5 | 4,935.53 MiB peak GPU allocation |

For this fixed request, GPU-path latency was about 1/22.8 of the 16-thread CPU path. The paths use different weight formats and arithmetic precision, so the ratio is not an isolated hardware speedup. CPU RSS and GPU framework allocation are different memory measures. The [sanitized JSON](benchmark-data/bit-jev-autodl-case-2026-09-27.json) contains each timing sample, hardware, checkpoint hashes, and limitations. The bilingual charts are generated from that file by [`test/render_checkpoint_public_figures.py`](../test/render_checkpoint_public_figures.py), without reading the local input request. This one-request development case is not a general speed or accuracy claim.

The checkpoint was trained on a multi-source decision set that includes Yelp review records. As of 2026-09-28, no written response had arrived to the permission request for derivative weights and metrics. The model card discloses this status; the maintainer chose to publish these sanitized aggregate measurements. No standalone open-weights license is specified for the checkpoint.

## CPU measurement

1. Record the exact source revision, model and pointer-head hashes, tokenizer revision, quantization format, compiler flags, and native binary hash.
2. Record CPU model, physical cores, allocated cores or cgroup quota, NUMA placement, thread count, batch size, operating system, and available RAM.
3. Describe the workload: number of requests and questions, options per question, encoded row lengths, and whether repeated shared state is included.
4. Separate process startup, model loading, tokenization, warmup, native compute, and end-to-end request wall time. Keep the model resident for steady-state measurements, and report cold-start results separately.
5. Run multiple rounds, retain individual timings, and report the sampling method for **peak process resident memory**. Model file size, RSS, and system memory are different measures.
6. Check finite logits, selected options, and numeric differences before comparing formats. Matching one answer is not an accuracy parity test.

The [input preparation script](../test/prepare_native_benchmark.py) can encode authorized JSONL requests. The [native format runner](../test/benchmark_native_formats.py) records per-request timing and process RSS; the [summary script](../test/summarize_cpu_format_benchmark.py) can compare two runs using the same input order. Put generated inputs and reports under the project root's `test/` directory and review source-data terms before sharing them.

## CPU versus GPU

For a hardware comparison, report each path's model format and arithmetic precision, warmup, synchronization, batch shape, model allocation or RSS definition, and measured interval. A CPU I2_S result and a mixed-precision GPU result describe **deployment paths**; they do not isolate the hardware effect. The [GPU measurement script](../test/benchmark_gpu.py) is an experimental forward benchmark, not a validated serving endpoint.

Core count can matter, but latency may also depend on memory bandwidth, clock speed, NUMA placement, input length, candidate count, and the number of question rows. Measure several thread counts on the same host and workload before claiming scaling. For the current native implementation, a multiquestion request repeats the shared state once per question.

## Accuracy and calibration

Freeze the model and evaluation protocol before scoring. Identify the dataset, source permissions, split, duplicate handling, option order, question types, and class counts. Separate development, locked test, and new-source evaluations. Publish per-type accuracy with denominators, numerical CPU/reference parity, and calibration checks when rights permit. Do not infer broad reliability from a small smoke input.

## Public result template

| Field | Record when an authorized release is measured |
| --- | --- |
| Artifact | Model, tokenizer, pointer-head and source hashes; redistribution basis |
| Workload | Dataset and split, source rights, request/question count, candidate counts, encoded input lengths |
| CPU path | Hardware allocation, quantization, threads, batch size, build flags |
| GPU path | Hardware, precision, synchronization, allocation method |
| Time | Individual warm and cold runs; mean/median/tail where supported; exact measured interval |
| Memory | File size, peak process RSS, GPU allocation, sampling method |
| Correctness | Finite outputs, option agreement, accuracy/calibration on a permitted evaluation set |

Every published chart links to its public measurement record. The bit-jev AutoDL case chart uses only the sanitized timing and memory values in the linked JSON; the local request and predictions are not part of the release.
