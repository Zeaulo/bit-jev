#!/usr/bin/env bash
# Run once on a fresh AutoDL instance (PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 image).
# Usage: bash scripts/setup_autodl.sh [workdir]   (default: /root/autodl-tmp/bit-jev)
set -euo pipefail

WORK=${1:-/root/autodl-tmp/bit-jev}
mkdir -p "$WORK"; cd "$WORK"

printf '== python/'; python -V
printf '== torch/'; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

# torch ships with the image; the rest pins what installs over it.
python -m pip install -U pip
python -m pip install \
  "transformers>=4.52.1,<5" "peft==0.14.0" "accelerate==1.3.0" \
  "datasets" "sentencepiece" "protobuf" "numpy<2" \
  "modelscope" "fastapi" "uvicorn"

# ModelScope mirrors HF for everything bit-jev needs; AutoDL cannot reach huggingface.co.
# bit_jev.hub switches to it automatically on instances that look like AutoDL; force it here so
# interactive shells behave the same way as scripts.
if ! grep -q BIT_JEV_HUB ~/.bashrc; then echo 'export BIT_JEV_HUB=modelscope' >> ~/.bashrc; fi
export BIT_JEV_HUB=modelscope

echo "done. Next: bash scripts/smoke_train.sh"
