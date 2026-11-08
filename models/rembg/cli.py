import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('U2NET_HOME', str(Path(__file__).resolve().parent / 'weights'))

from PIL import Image
from rembg import new_session, remove


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='bria-rmbg')
    parser.add_argument('--only-mask', action='store_true')
    parser.add_argument('--alpha-matting', action='store_true')
    parser.add_argument('--alpha-matting-foreground-threshold', type=int, default=240)
    parser.add_argument('--alpha-matting-background-threshold', type=int, default=10)
    parser.add_argument('--alpha-matting-erode-size', type=int, default=10)
    parser.add_argument('--post-process-mask', action='store_true')
    parser.add_argument('--decontaminate', action='store_true')
    parser.add_argument('--vitmatte', action='store_true')
    parser.add_argument('--vitmatte-model', default=None)
    parser.add_argument('--putalpha', action='store_true')
    parser.add_argument('--sam-prompt', default=None)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--bgcolor', default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    if args.threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)

    kwargs: dict = {'session': new_session(args.model)}
    if args.only_mask:
        kwargs['only_mask'] = True
    if args.post_process_mask:
        kwargs['post_process_mask'] = True
    if args.decontaminate:
        kwargs['decontaminate'] = True
    if args.vitmatte:
        kwargs['vitmatte'] = True
        if args.vitmatte_model is not None:
            kwargs['vitmatte_model'] = args.vitmatte_model
    if args.putalpha:
        kwargs['putalpha'] = True
    if args.sam_prompt is not None:
        kwargs['sam_prompt'] = json.loads(args.sam_prompt)
    if args.alpha_matting:
        kwargs['alpha_matting'] = True
        kwargs['alpha_matting_foreground_threshold'] = args.alpha_matting_foreground_threshold
        kwargs['alpha_matting_background_threshold'] = args.alpha_matting_background_threshold
        kwargs['alpha_matting_erode_size'] = args.alpha_matting_erode_size
    if args.bgcolor is not None:
        kwargs['bgcolor'] = [int(v) for v in args.bgcolor.split(',')]

    with Image.open(args.input) as image:
        result = remove(image, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)

    print(json.dumps({
        'status': 'ok',
        'output': str(args.output),
        'total_seconds': round(time.monotonic() - started, 1),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
