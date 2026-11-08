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
    parser.add_argument('--compute-type', default='int8')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--diarize', action='store_true')
    parser.add_argument('--min-speakers', type=int, default=None)
    parser.add_argument('--max-speakers', type=int, default=None)
    parser.add_argument('--num-speakers', type=int, default=None)
    parser.add_argument('--fill-nearest', action='store_true')
    parser.add_argument('--beam-size', type=int, default=None)
    parser.add_argument('--best-of', type=int, default=None)
    parser.add_argument('--patience', type=float, default=None)
    parser.add_argument('--length-penalty', type=float, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=None)
    parser.add_argument('--log-prob-threshold', type=float, default=None)
    parser.add_argument('--no-speech-threshold', type=float, default=None)
    parser.add_argument('--compression-ratio-threshold', type=float, default=None)
    parser.add_argument('--condition-on-previous-text', default=None)
    parser.add_argument('--prompt-reset-on-temperature', type=float, default=None)
    parser.add_argument('--temperatures', default=None)
    parser.add_argument('--prefix', default=None)
    parser.add_argument('--suppress-blank', default=None)
    parser.add_argument('--suppress-tokens', default=None)
    parser.add_argument('--prepend-punctuations', default=None)
    parser.add_argument('--append-punctuations', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument('--clip-timestamps', default=None)
    parser.add_argument('--hallucination-silence-threshold', type=float, default=None)
    parser.add_argument('--suppress-numerals', default=None)
    return parser.parse_args()


def to_bool(value: str) -> bool:
    return value == 'true'


def to_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(',') if part]


def to_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(',') if part]


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    asr_options = {}
    if args.initial_prompt is not None:
        asr_options['initial_prompt'] = args.initial_prompt
    if args.hotwords is not None:
        asr_options['hotwords'] = args.hotwords
    if args.beam_size is not None:
        asr_options['beam_size'] = args.beam_size
    if args.best_of is not None:
        asr_options['best_of'] = args.best_of
    if args.patience is not None:
        asr_options['patience'] = args.patience
    if args.length_penalty is not None:
        asr_options['length_penalty'] = args.length_penalty
    if args.repetition_penalty is not None:
        asr_options['repetition_penalty'] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        asr_options['no_repeat_ngram_size'] = args.no_repeat_ngram_size
    if args.log_prob_threshold is not None:
        asr_options['log_prob_threshold'] = args.log_prob_threshold
    if args.no_speech_threshold is not None:
        asr_options['no_speech_threshold'] = args.no_speech_threshold
    if args.compression_ratio_threshold is not None:
        asr_options['compression_ratio_threshold'] = args.compression_ratio_threshold
    if args.condition_on_previous_text is not None:
        asr_options['condition_on_previous_text'] = to_bool(args.condition_on_previous_text)
    if args.prompt_reset_on_temperature is not None:
        asr_options['prompt_reset_on_temperature'] = args.prompt_reset_on_temperature
    if args.temperatures is not None:
        asr_options['temperatures'] = to_float_list(args.temperatures)
    if args.prefix is not None:
        asr_options['prefix'] = args.prefix
    if args.suppress_blank is not None:
        asr_options['suppress_blank'] = to_bool(args.suppress_blank)
    if args.suppress_tokens is not None:
        asr_options['suppress_tokens'] = to_int_list(args.suppress_tokens)
    if args.prepend_punctuations is not None:
        asr_options['prepend_punctuations'] = args.prepend_punctuations
    if args.append_punctuations is not None:
        asr_options['append_punctuations'] = args.append_punctuations
    if args.max_new_tokens is not None:
        asr_options['max_new_tokens'] = args.max_new_tokens
    if args.clip_timestamps is not None:
        asr_options['clip_timestamps'] = to_float_list(args.clip_timestamps)
    if args.hallucination_silence_threshold is not None:
        asr_options['hallucination_silence_threshold'] = args.hallucination_silence_threshold
    if args.suppress_numerals is not None:
        asr_options['suppress_numerals'] = to_bool(args.suppress_numerals)

    load_kwargs = {'device': 'cpu', 'compute_type': args.compute_type, 'asr_options': asr_options or None}
    if args.threads is not None:
        load_kwargs['threads'] = args.threads

    model = whisperx.load_model(args.model_size, **load_kwargs)
    audio = whisperx.load_audio(str(args.input))
    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language, task=args.task)
    language = result['language']

    segments = result['segments']
    word_segments = []
    aligned = False
    if segments:
        try:
            align_model, align_metadata = whisperx.load_align_model(language_code=language, device='cpu')
            result = whisperx.align(segments, align_model, align_metadata, audio, 'cpu')
            segments = result['segments']
            word_segments = result.get('word_segments', [])
            aligned = True
        except Exception:
            aligned = False

    diarized = False
    if args.diarize and aligned:
        pipeline = DiarizationPipeline(token=os.environ.get('HF_TOKEN'))
        diarization = pipeline(
            str(args.input),
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarization, result, fill_nearest=args.fill_nearest)
        segments = result['segments']
        word_segments = result.get('word_segments', [])
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
                'word_segments': word_segments,
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
