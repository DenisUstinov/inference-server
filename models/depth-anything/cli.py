import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "weights"))

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = DepthAnythingForDepthEstimation.from_pretrained(MODEL_ID)

    with Image.open(args.input) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            depth = model(**inputs).predicted_depth[0]

    depth_array = depth.cpu().numpy().astype("float32")
    span = float(depth_array.max() - depth_array.min()) + 1e-8
    normalized = (depth_array - float(depth_array.min())) / span
    depth_16 = (normalized * 65535).astype(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_16).save(args.output)

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "total_seconds": round(time.monotonic() - started, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
