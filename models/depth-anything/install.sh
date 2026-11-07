#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

export HF_HOME="$MODEL_DIR/weights"

uv sync

uv run python - <<'PY'
import torch
from PIL import Image
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

model_id = "depth-anything/Depth-Anything-V2-Small-hf"
processor = AutoImageProcessor.from_pretrained(model_id)
model = DepthAnythingForDepthEstimation.from_pretrained(model_id)
inputs = processor(images=Image.new("RGB", (64, 64), (127, 127, 127)), return_tensors="pt")
with torch.no_grad():
    depth = model(**inputs).predicted_depth
assert depth.shape[0] == 1
PY
