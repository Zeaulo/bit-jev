# bit-jev-2b-distilled

> Hugging Face release draft. This is a local review copy until written permission covering the Yelp-derived checkpoint and its aggregate measurements is received.

bit-jev-2b-distilled is a structured decision model built on a BitNet backbone. It reads shared state plus one or more questions and scores caller-supplied options. It returns structured decisions rather than autoregressively generating answer text.

The training path is BitNet BF16 backbone → LoRA and pointer-head fine-tuning → teacher-logit export → full-student distillation → I2_S GGUF plus a float32 pointer head.

The final Hub repository should contain the CPU GGUF, pointer-head sidecar, tokenizer/configuration files, a SHA-256 manifest, this model card, and a runnable example. It must not contain raw Yelp review records.

The AutoDL case study is a single fixed request with 703 input tokens and 77 options. CPU I2_S native inference and GPU FP16 mixed-precision inference use different formats, so their latency ratio is a deployment-path comparison rather than an isolated hardware speedup. The raw local reports remain under `test/release/` pending permission.

Following Kev's reporting style, the release should separate trained-source and new-source evaluation, report per-question-type accuracy and calibration, and bind every performance number to an immutable checkpoint, input shape, hardware, precision, warmup and memory definition.

