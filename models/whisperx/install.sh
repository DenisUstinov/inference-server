#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

export HF_HOME="$MODEL_DIR/weights"
export TORCH_HOME="$MODEL_DIR/weights/torch"
export NLTK_DATA="$MODEL_DIR/weights/nltk"

mkdir -p "$NLTK_DATA"
chmod go-w "$MODEL_DIR" "$MODEL_DIR/weights" "$NLTK_DATA"

uv sync

uv run python - <<'PY'
import os

import nltk
import whisperx
from whisperx.diarize import DiarizationPipeline

nltk.download("punkt_tab")
nltk.download("punkt")
assert nltk.data.find("tokenizers/punkt_tab")
assert nltk.tokenize.word_tokenize("test sentence") == ["test", "sentence"]

for size in ("tiny", "base", "small", "medium", "large-v3", "turbo"):
    whisperx.load_model(size, device="cpu", compute_type="int8")

whisperx.load_align_model(language_code="en", device="cpu")
whisperx.load_align_model(language_code="ru", device="cpu")

DiarizationPipeline(token=os.environ.get("HF_TOKEN"))
PY
