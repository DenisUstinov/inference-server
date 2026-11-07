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

SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action='append', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-audio', type=Path, default=None)
    parser.add_argument('--reference-transcript', default=None)
    parser.add_argument('--instruct', default=None)
    parser.add_argument('--language', default=None)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--speed', type=float, default=None)
    parser.add_argument('--normalize-text', action='store_true')
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


def load_reference_audio(path: Path) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(path))
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    return waveform


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
    started = time.monotonic()

    model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)
    load_seconds = time.monotonic() - started

    common: dict = {}
    if args.reference_audio is not None:
        common['prompt_wav'] = load_reference_audio(args.reference_audio)
        common['prompt_text'] = args.reference_transcript or ''
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
