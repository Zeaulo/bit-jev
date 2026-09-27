#!/usr/bin/env bash
# One-time: a SEPARATE venv for the kev-9B teacher (kev needs transformers>=5.17 +
# flash-linear-attention; bit-jev's BitNet path pins transformers 4.5x -- they cannot coexist).
set -euo pipefail

VENV=${1:-/root/autodl-tmp/kev-venv}
if [ ! -f "$VENV/bin/python" ]; then
  python -m venv "$VENV"
fi
"$VENV/bin/pip" install -U pip
# kev's own requirements (from its pyproject): transformers>=5.17, peft>=0.21, plus fla for the
# Qwen3.5 hybrid backbone on CUDA. torch comes from the base env via --system-site-packages? No:
# venv is isolated; install the CUDA torch wheel explicitly.
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install "transformers>=5.17" "peft>=0.21" flash-linear-attention "triton>=3.7.1" \
  "datasets" "sentencepiece" "protobuf" "numpy<2" "huggingface_hub" accelerate
export HF_ENDPOINT=https://hf-mirror.com
"$VENV/bin/python" -c "import transformers, torch; print('kev venv ready:', transformers.__version__, torch.__version__)"
