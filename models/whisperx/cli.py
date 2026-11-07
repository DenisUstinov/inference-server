import argparse
import json
import os
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

os.environ.setdefault('HF_HOME', str(MODEL_DIR / 'weights'))
os.environ.setdefault('TORCH_HOME', str(MODEL_DIR / 'weights' / 'torch'))
os.environ.setdefault('NLTK_DATA', str(MODEL_DIR / 'weights' / 'nltk'))

import whisperx
from whisperx.diarize import DiarizationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--language', default=None)
    parser.add_argument('--task', default='transcribe')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--initial-prompt', default=None)
    parser.add_argument('--hotwords', default=None)
    parser.add_argument('--model-size', default='large-v3')
    parser.add_argument('--diarize', action='store_true')
    parser.add_argument('--min-speakers', type=int, default=None)
    parser.add_argument('--max-speakers', type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    asr_options = {}
    if args.initial_prompt is not None:
        asr_options['initial_prompt'] = args.initial_prompt
    if args.hotwords is not None:
        asr_options['hotwords'] = args.hotwords

    model = whisperx.load_model(
        args.model_size,
        device='cpu',
        compute_type='int8',
        asr_options=asr_options or None,
    )
    audio = whisperx.load_audio(str(args.input))
    result = model.transcribe(
        audio,
        batch_size=args.batch_size,
        language=args.language,
        task=args.task,
    )
    language = result['language']

    segments = result['segments']
    aligned = False
    if segments:
        try:
            align_model, align_metadata = whisperx.load_align_model(language_code=language, device='cpu')
            result = whisperx.align(segments, align_model, align_metadata, audio, 'cpu')
            segments = result['segments']
            aligned = True
        except Exception:
            aligned = False

    diarized = False
    if args.diarize and aligned:
        pipeline = DiarizationPipeline(token=os.environ.get('HF_TOKEN'))
        diarization = pipeline(str(args.input), min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        result = whisperx.assign_word_speakers(diarization, result)
        segments = result['segments']
        diarized = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                'status': 'ok',
                'task': args.task,
                'language': language,
                'text': ' '.join(segment['text'].strip() for segment in segments),
                'segments': segments,
                'aligned': aligned,
                'diarized': diarized,
                'total_seconds': round(time.monotonic() - started, 1),
            },
            ensure_ascii=False,
            default=str,
        ),
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
