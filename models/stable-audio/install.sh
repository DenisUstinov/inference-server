#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$MODEL_DIR/project"
REPO_URL="${REPO_URL:-https://github.com/Stability-AI/stable-audio-3.git}"
COMMIT="790793b89ddceea612be9ce42ff6830d18f6fa57"
LITERT_PACKAGE="ai-edge-litert-nightly"
LITERT_VERSION="2.4.0.dev20261105"

if [[ ! -d "$PROJECT/.git" ]]; then
    git clone "$REPO_URL" "$PROJECT"
fi
git -C "$PROJECT" checkout --quiet --detach "$COMMIT"

cd "$PROJECT"

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi

cd "$PROJECT/optimized/tflite"

if [[ ! -e .venv ]]; then
    ln -s ../../.venv .venv
fi

export HF_HOME="$MODEL_DIR/weights"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$PROJECT/optimized/tflite/models/tflite/lora_cache"

./install.sh --download medium

uv pip uninstall --python "$PROJECT/.venv/bin/python" ai-edge-litert || true
uv pip install --python "$PROJECT/.venv/bin/python" --quiet --reinstall-package "$LITERT_PACKAGE" "$LITERT_PACKAGE==$LITERT_VERSION"

LITERT_VERSION="$LITERT_VERSION" .venv/bin/python - <<'PY'
import importlib.metadata
import os

from ai_edge_litert import interpreter
from huggingface_hub import hf_hub_download

assert interpreter is not None
assert importlib.metadata.version("ai-edge-litert-nightly") == os.environ["LITERT_VERSION"]

files = [
    "tflite/sa3-m/dit_fp32.tflite",
    "tflite/sa3-m/dit_w8a8.tflite",
    "tflite/sa3-sm-music/dit_fp32.tflite",
    "tflite/sa3-sm-music/dit_w8a8.tflite",
    "tflite/sa3-sm-sfx/dit_fp32.tflite",
    "tflite/sa3-sm-sfx/dit_w8a8.tflite",
    "tflite/same-l/enc_w8a8.tflite",
    "tflite/same-l/dec_w8a8.tflite",
    "tflite/same-l/enc_fp32.tflite",
    "tflite/same-l/dec_fp32.tflite",
    "tflite/same-s/enc_w8a8.tflite",
    "tflite/same-s/dec_w8a8.tflite",
    "tflite/same-s/enc_fp32.tflite",
    "tflite/same-s/dec_fp32.tflite",
    "tflite/t5gemma/encoder_fp16.tflite",
    "tokenizer.model",
]
for name in files:
    hf_hub_download("stabilityai/stable-audio-3-optimized", name)
PY

.venv/bin/python scripts/sa3_tflite.py --prompt "install smoke" --dit medium --dit-precision fp32 --decoder same-l --decoder-precision w8a8 --seconds 10 --seed 42 --out /tmp/sa3-install-smoke.wav

warm() {
    .venv/bin/python scripts/sa3_tflite.py --prompt "cache warm" --seed 1 --seconds 1 --out /tmp/sa3-warm.wav "$@"
}

warm_a2a() {
    .venv/bin/python scripts/sa3_tflite.py --prompt "cache warm" --seed 1 --seconds 10 \
        --dit sm-music --dit-precision w8a8 \
        --init-audio /tmp/sa3-install-smoke.wav --init-noise-level 0.5 \
        --out /tmp/sa3-warm.wav "$@"
}

warm --dit medium --dit-precision w8a8 --decoder same-s --decoder-precision w8a8
warm --dit sm-music --dit-precision w8a8 --decoder same-s --decoder-precision w8a8
warm --dit sm-music --dit-precision fp32 --decoder same-s --decoder-precision w8a8
warm --dit sm-sfx --dit-precision w8a8 --decoder same-s --decoder-precision w8a8
warm --dit sm-sfx --dit-precision fp32 --decoder same-s --decoder-precision w8a8
warm --dit medium --dit-precision w8a8 --decoder same-l --decoder-precision fp32
warm --dit medium --dit-precision w8a8 --decoder same-s --decoder-precision fp32
warm_a2a --decoder same-s --decoder-precision w8a8 --encoder-precision w8a8
warm_a2a --decoder same-s --decoder-precision w8a8 --encoder-precision fp32
warm_a2a --decoder same-l --decoder-precision w8a8 --encoder-precision w8a8
warm_a2a --decoder same-l --decoder-precision w8a8 --encoder-precision fp32

rm -f /tmp/sa3-install-smoke.wav /tmp/sa3-warm.wav
