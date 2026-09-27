# v0.7.7 source, bilingual highlights and public-base measurement release checklist

This checklist covers **public source and independently measured Microsoft BitNet base-model figures**. It does not authorize publication of the local trained bit-jev checkpoint, I2_S export, model output, or checkpoint-derived benchmark reports.

## Public source boundary

- [ ] Confirm the source version, tag, `versions/update.log`, and project overview describe the same source-only release.
- [ ] Keep trained bit-jev weights, GGUF exports, pointer-head weights, tokenizer copies from a trained package, teacher logits, training/evaluation records, model outputs, derived reports, and performance charts out of the public source tree. The Microsoft public-base JSON and charts must have verified source revision and model SHA-256.
- [ ] Scan the **entire history being pushed**, not just the final file list. Removed files in an earlier reachable commit remain publicly accessible. Use a clean public root commit or an equivalent history rewrite after checking the contents.
- [ ] Scan tracked files for credentials, private keys, workstation paths, and output that could reconstruct restricted source records. Confirm `git ls-files` and inspect the public commit before pushing.
- [ ] Keep root [LICENSE](../LICENSE) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), including required upstream notices. Verify upstream source licenses against the pinned revisions.
- [ ] Ensure [README](../README.md), [CPU quick start](CPU_QUICKSTART.md), [model status](MODEL_CARD.md), and [benchmark protocol](BENCHMARKS.md) contain no current-checkpoint results, live weight claims, or dead release-asset download instructions.

## Source reproducibility

- [ ] Clone the public repository in a fresh directory. Run `python test/bootstrap_bitnet.py` and `python test/bootstrap_bitnet.py --check`; record the exact pinned upstream commits and patch checks.
- [ ] Build `bit-jev-cpu` from that clone with the documented CMake command. Record compiler, CMake, operating system, CPU architecture, and bootstrap network limitations.
- [ ] Confirm the native and Python interfaces accept the documented request shape. Any actual inference check must use a separately authorized compatible model; do not turn a private local checkpoint into a public fixture.
- [ ] Verify that all public Markdown links, the [Chinese highlights](figures/model-highlights.zh-CN.svg), [English highlights](figures/model-highlights.en.svg), [detailed framework](figures/model-framework.svg), [public-base speed chart](figures/public-base-speed.svg), and [memory chart](figures/public-base-memory.svg) render on GitHub.

## Publication

- [ ] Update the existing public `Zeaulo/bit-jev` repository from the reviewed history, push `v0.7.7`, and open it while signed out.
- [ ] If a GitHub release is created, label the bit-jev checkpoint **withheld**; link the public Microsoft base-model benchmark data and attach no model archive.
- [ ] Publish the [X launch copy](X_LAUNCH.md) only after the repository URL resolves and the post labels the speed chart as a Microsoft base-model measurement.
- [ ] Invite architecture feedback, build reports, and independent benchmarks produced with artifacts and data that contributors may share.

## Separate gate for any future trained model

Before distributing a model or checkpoint-derived metrics, document the precise base and teacher revisions, training/evaluation sources, rights to publish the derivative and results, release license, model hashes, clean-machine package test, numerical parity, accuracy/calibration scope, timing method, and memory units. In particular, resolve the local Yelp training-data issue; the [Yelp dataset card](https://huggingface.co/datasets/Yelp/yelp_review_full) links to separate [dataset terms](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf). The present source-only release leaves this gate open.
