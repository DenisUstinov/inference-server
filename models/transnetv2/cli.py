import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

os.environ.setdefault('HF_HOME', str(MODEL_DIR / 'weights'))

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(MODEL_DIR / 'project' / 'inference-pytorch'))
from transnetv2_pytorch import TransNetV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


def extract_frames(path: Path) -> np.ndarray:
    completed = subprocess.run(
        [
            'ffmpeg', '-loglevel', 'error',
            '-i', str(path),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '48x27',
            'pipe:1',
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, 27, 48, 3)


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
    binary = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(binary):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return [[0, len(binary) - 1]]
    return scenes


def predict(frames: np.ndarray, model: TransNetV2) -> np.ndarray:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(frames), 100):
            chunk = frames[start:start + 100]
            valid = len(chunk)
            if valid < 100:
                padding = np.zeros((100 - valid, 27, 48, 3), dtype=np.uint8)
                chunk = np.concatenate([chunk, padding])
            tensor = torch.from_numpy(chunk.copy()).unsqueeze(0)
            single_frame_pred, _ = model(tensor)
            predictions.append(
                torch.sigmoid(single_frame_pred).cpu().numpy()[0, :valid].reshape(-1)
            )
    return np.concatenate(predictions) if predictions else np.array([])


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    weights_dir = Path(snapshot_download('MiaoshouAI/transnetv2-pytorch-weights'))
    weights_path = next(weights_dir.rglob('*.pth'))

    frames = extract_frames(args.input)

    model = TransNetV2()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    all_predictions = predict(frames, model)
    scenes = predictions_to_scenes(all_predictions, args.threshold) if len(all_predictions) else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                'status': 'ok',
                'num_frames': int(len(frames)),
                'scenes': scenes,
                'total_seconds': round(time.monotonic() - started, 1),
            }
        ),
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
