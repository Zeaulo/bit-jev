# Benchmark protocol for future authorized artifacts

This source-only release contains **no public measurements for the project's trained checkpoint**. Its weights, model outputs, and checkpoint-derived accuracy, speed, and memory reports are withheld pending training-data rights review. The scripts in [test/](../test/) are measurement tools; their presence does not make a result reproducible without an authorized compatible model and input set.

bit-jev scores explicit options after reading the input. It does not decode answer text token by token, so generated-output tokens per second is not an appropriate throughput measure. Report **request latency** and, if useful, **input tokens per second** with the token-count definition stated.

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

A future chart should link to the exact input and report used to produce it. No chart in this source-only release represents a publicly distributed bit-jev checkpoint.
