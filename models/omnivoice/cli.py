import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent / 'weights'))

import numpy as np
import torch
import torchaudio
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.common import fix_random_seed

SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action='append', default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--reference-audio', type=Path, default=None)
    parser.add_argument('--reference-transcript', default=None)
    parser.add_argument('--voice-prompt', type=Path, default=None)
    parser.add_argument('--prompt-out', type=Path, default=None)
    parser.add_argument('--instruct', default=None)
    parser.add_argument('--language', default=None)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--speed', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--normalize-text', action='store_true')
    parser.add_argument('--preprocess-prompt', default=None)
    parser.add_argument('--postprocess-output', default=None)
    parser.add_argument('--num-step', type=int, default=None)
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--t-shift', type=float, default=None)
    parser.add_argument('--layer-penalty-factor', type=float, default=None)
    parser.add_argument('--position-temperature', type=float, default=None)
    parser.add_argument('--class-temperature', type=float, default=None)
    parser.add_argument('--denoise', default=None)
    parser.add_argument('--audio-chunk-duration', type=float, default=None)
    parser.add_argument('--audio-chunk-threshold', type=float, default=None)
    parser.add_argument('--pad-duration', type=float, default=None)
    parser.add_argument('--fade-duration', type=float, default=None)
    return parser.parse_args()


def to_wav_tensor(audio: object) -> torch.Tensor:
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.detach().to(dtype=torch.float32, device='cpu')
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    return audio


def main() -> int:
    args = parse_args()
    create_mode = args.prompt_out is not None
    if create_mode:
        if args.reference_audio is None:
            raise SystemExit('--reference-audio is required in prompt creation mode')
    elif not args.text or args.output is None:
        raise SystemExit('--text and --output are required in generation mode')

    started = time.monotonic()

    if args.seed is not None:
        fix_random_seed(args.seed)

    model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)
    load_seconds = time.monotonic() - started

    preprocess_prompt = None
    if args.preprocess_prompt is not None:
        preprocess_prompt = args.preprocess_prompt == 'true'

    if create_mode:
        factory_kwargs = {}
        if preprocess_prompt is not None:
            factory_kwargs['preprocess_prompt'] = preprocess_prompt
        prompt = model.create_voice_clone_prompt(
            str(args.reference_audio),
            ref_text=args.reference_transcript,
            **factory_kwargs,
        )
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        prompt.save(str(args.prompt_out))
        print(json.dumps({
            'status': 'ok',
            'prompt_out': str(args.prompt_out),
            'total_seconds': round(time.monotonic() - started, 1),
        }))
        return 0

    common: dict = {}
    if args.voice_prompt is not None:
        common['voice_clone_prompt'] = VoiceClonePrompt.load(str(args.voice_prompt))
    elif args.reference_audio is not None:
        factory_kwargs = {}
        if preprocess_prompt is not None:
            factory_kwargs['preprocess_prompt'] = preprocess_prompt
        common['voice_clone_prompt'] = model.create_voice_clone_prompt(
            str(args.reference_audio),
            ref_text=args.reference_transcript,
            **factory_kwargs,
        )
    if args.instruct is not None:
        common['instruct'] = args.instruct
    if args.language is not None:
        common['language'] = args.language
    if args.duration is not None:
        common['duration'] = args.duration
    if args.speed is not None:
        common['speed'] = args.speed
    if args.normalize_text:
        common['normalize_text'] = True
    if preprocess_prompt is not None:
        common['preprocess_prompt'] = preprocess_prompt
    if args.postprocess_output is not None:
        common['postprocess_output'] = args.postprocess_output == 'true'
    if args.num_step is not None:
        common['num_step'] = args.num_step
    if args.guidance_scale is not None:
        common['guidance_scale'] = args.guidance_scale
    if args.t_shift is not None:
        common['t_shift'] = args.t_shift
    if args.layer_penalty_factor is not None:
        common['layer_penalty_factor'] = args.layer_penalty_factor
    if args.position_temperature is not None:
        common['position_temperature'] = args.position_temperature
    if args.class_temperature is not None:
        common['class_temperature'] = args.class_temperature
    if args.denoise is not None:
        common['denoise'] = args.denoise == 'True'
    if args.audio_chunk_duration is not None:
        common['audio_chunk_duration'] = args.audio_chunk_duration
    if args.audio_chunk_threshold is not None:
        common['audio_chunk_threshold'] = args.audio_chunk_threshold
    if args.pad_duration is not None:
        common['pad_duration'] = args.pad_duration
    if args.fade_duration is not None:
        common['fade_duration'] = args.fade_duration

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    generate_started = time.monotonic()
    for index, text in enumerate(args.text):
        audio = model.generate(text=text, **common)
        if len(args.text) == 1:
            target = args.output
        else:
            target = args.output.parent / f'{args.output.stem}-{index:03d}{args.output.suffix}'
        torchaudio.save(str(target), to_wav_tensor(audio), SAMPLE_RATE)
        outputs.append(str(target))
    generate_seconds = time.monotonic() - generate_started

    print(json.dumps({
        'status': 'ok',
        'outputs': outputs,
        'load_seconds': round(load_seconds, 1),
        'generate_seconds': round(generate_seconds, 1),
        'total_seconds': round(time.monotonic() - started, 1),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
