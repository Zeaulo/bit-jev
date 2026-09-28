# v0.10.0 source, Hugging Face model and benchmark publication record

This checklist records the **GitHub code release**, the Windows x64 AVX2 precompiled CPU runner, independent Microsoft BitNet base-model figures, and the separate [Hugging Face bit-jev model package](https://huggingface.co/jinghao1632/bit-jev-2b-distilled). GitHub contains code and the small CPU executable; the GGUF model remains on Hugging Face.

## Public source boundary

- [ ] Confirm the source version, tag, `versions/update.log`, and project overview describe the same GitHub source release and link to the Hub package.
- [ ] Keep trained bit-jev weights, GGUF exports, pointer-head weights, tokenizer copies from the trained package, teacher logits, training/evaluation records, and model outputs out of the GitHub tree. Publish only the sanitized AutoDL aggregate JSON and bilingual charts; keep the Microsoft public-base JSON tied to its verified source revision and model SHA-256.
- [ ] Scan the **entire history being pushed**, not just the final file list. Removed files in an earlier reachable commit remain publicly accessible. Use a clean public root commit or an equivalent history rewrite after checking the contents.
- [ ] Scan tracked files for credentials, private keys, workstation paths, and output that could reconstruct restricted source records. Confirm `git ls-files` and inspect the public commit before pushing.
- [ ] Keep root [LICENSE](../LICENSE) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), including required upstream notices. Verify upstream source licenses against the pinned revisions.
- [ ] Ensure [README](../README.md), [CPU quick start](CPU_QUICKSTART.md), [model status](MODEL_CARD.md), and [benchmark protocol](BENCHMARKS.md) link to the same Hub repository and label the single-question case accurately.

## Source reproducibility

- [ ] Clone the public repository in a fresh directory. Run `python test/bootstrap_bitnet.py` and `python test/bootstrap_bitnet.py --check`; record the exact pinned upstream commits and patch checks.
- [ ] Build `bit-jev-cpu` from that clone with the documented CMake command. Record compiler, CMake, operating system, CPU architecture, and bootstrap network limitations.
- [ ] Confirm the native and Python interfaces accept the documented request shape. Any actual inference check must use a separately authorized compatible model; do not turn a private local checkpoint into a public fixture.
- [ ] Verify that all public Markdown links, the [Chinese highlights](figures/model-highlights.zh-CN.svg), [English highlights](figures/model-highlights.en.svg), [detailed framework](figures/model-framework.svg), [public-base speed chart](figures/public-base-speed.svg), and [memory chart](figures/public-base-memory.svg) render on GitHub.

## Publication

- [ ] Update the existing public `Zeaulo/bit-jev` repository from the reviewed history, push the current version, and open it while signed out.
- [ ] If a GitHub release is created, attach no model archive; link the separate Hugging Face model package and both benchmark data records.
- [ ] Publish the [X launch copy](X_LAUNCH.md) only after the repository URL resolves and the post labels the speed chart as a Microsoft base-model measurement.
- [ ] Invite architecture feedback, build reports, and independent benchmarks produced with artifacts and data that contributors may share.

## Rights and evaluation status for the published trained model

The model and its sanitized single-question measurements are published. The permission request to Yelp remains unanswered as of 2026-09-28; the [Yelp dataset card](https://huggingface.co/datasets/Yelp/yelp_review_full) links to separate [dataset terms](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf). The Hub model card records the unresolved status and does not assign an open-weights license. The package hash list, clean-machine inference, and any future held-out accuracy/calibration report should be updated when independently verified; current public performance claims remain limited to the labeled AutoDL case and the separate Microsoft base-model benchmark.
