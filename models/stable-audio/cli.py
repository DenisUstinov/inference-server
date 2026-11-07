import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = MODEL_DIR / 'project' / 'optimized' / 'tflite'

os.environ.setdefault('HF_HOME', str(MODEL_DIR / 'weights'))
os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dit', default='medium')
    parser.add_argument('--dit-precision', default='fp32')
    parser.add_argument('--decoder', default='same-l')
    parser.add_argument('--decoder-precision', default='w8a8')
    parser.add_argument('--encoder-precision', default='w8a8')
    parser.add_argument('--cfg', type=float, default=None)
    parser.add_argument('--negative-prompt', default=None)
    parser.add_argument('--init-audio', type=Path, default=None)
    parser.add_argument('--init-noise-level', type=float, default=None)
    parser.add_argument('--inpaint-range', default=None)
    parser.add_argument('--max-rung', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--apg', type=float, default=None)
    parser.add_argument('--lora', nargs='+', default=None)
    parser.add_argument('--lora-strength', type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    command = [
        sys.executable,
        str(RUNTIME_DIR / 'scripts' / 'sa3_tflite.py'),
        '--prompt', args.text,
        '--dit', args.dit,
        '--dit-precision', args.dit_precision,
        '--decoder', args.decoder,
        '--decoder-precision', args.decoder_precision,
        '--encoder-precision', args.encoder_precision,
        '--seconds', str(args.seconds),
        '--out', str(args.output),
    ]
    if args.seed is not None:
        command.extend(['--seed', str(args.seed)])
    if args.cfg is not None:
        command.extend(['--cfg', str(args.cfg)])
    if args.negative_prompt is not None:
        command.extend(['--negative-prompt', args.negative_prompt])
    if args.init_audio is not None:
        command.extend(['--init-audio', str(args.init_audio)])
    if args.init_noise_level is not None:
        command.extend(['--init-noise-level', str(args.init_noise_level)])
    if args.inpaint_range is not None:
        command.extend(['--inpaint-range', args.inpaint_range])
    if args.max_rung is not None:
        command.extend(['--max-rung', str(args.max_rung)])
    if args.steps is not None:
        command.extend(['--steps', str(args.steps)])
    if args.apg is not None:
        command.extend(['--apg', str(args.apg)])
    if args.lora is not None:
        command.append('--lora')
        command.extend(args.lora)
        if args.lora_strength is not None:
            command.extend(['--lora-strength', str(args.lora_strength)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=RUNTIME_DIR, check=False)

    if completed.returncode != 0 or not args.output.exists():
        return completed.returncode or 1

    print(json.dumps({
        'status': 'ok',
        'output': str(args.output),
        'total_seconds': round(time.monotonic() - started, 1),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
