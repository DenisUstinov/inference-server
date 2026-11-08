#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$MODEL_DIR/project"
REPO_URL="${REPO_URL:-https://github.com/soCzech/TransNetV2.git}"
COMMIT="85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed"

if [[ ! -d "$PROJECT/.git" ]]; then
    git clone "$REPO_URL" "$PROJECT"
fi
git -C "$PROJECT" checkout --quiet --detach "$COMMIT"

cd "$MODEL_DIR"

export HF_HOME="$MODEL_DIR/weights"

uv sync

uv run python - <<'PY'
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, "project/inference-pytorch")
from transnetv2_pytorch import TransNetV2

weights_dir = Path(snapshot_download("MiaoshouAI/transnetv2-pytorch-weights"))
weights_path = next(weights_dir.rglob("*.pth"))

model = TransNetV2()
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()
dummy = np.zeros((1, 100, 27, 48, 3), dtype=np.uint8)
with torch.no_grad():
    single_frame_pred, all_frame_pred = model(torch.from_numpy(dummy))
assert single_frame_pred.shape[0] == 1
PY
