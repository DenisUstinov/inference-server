#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

export U2NET_HOME="$MODEL_DIR/weights"

uv sync

mkdir -p "$U2NET_HOME"
uv run python - <<'PY'
from PIL import Image
from rembg import new_session, remove

image = Image.new("RGB", (64, 64), (127, 127, 127))
sessions = [
    "bria-rmbg",
    "birefnet-cod",
    "birefnet-dis",
    "birefnet-general",
    "birefnet-general-fp16",
    "birefnet-general-lite",
    "birefnet-hrsod",
    "birefnet-massive",
    "birefnet-portrait",
    "isnet-anime",
    "isnet-general-use",
    "sam",
    "silueta",
    "u2net",
    "u2net_cloth_seg",
    "u2net_human_seg",
    "u2netp",
]
for name in sessions:
    remove(image, session=new_session(name))
for variant in ("small-distinctions-646", "small-composition-1k", "base-distinctions-646", "base-composition-1k"):
    remove(image, session=new_session("bria-rmbg"), vitmatte=True, vitmatte_model=variant)
PY
