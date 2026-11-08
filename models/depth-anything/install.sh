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

model_ids = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
]
for model_id in model_ids:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = DepthAnythingForDepthEstimation.from_pretrained(model_id)
    inputs = processor(images=Image.new("RGB", (64, 64), (127, 127, 127)), return_tensors="pt")
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    assert depth.shape[0] == 1
    print(f"smoke ok: {model_id}")
PY
