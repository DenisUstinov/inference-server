import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import rpyc
from rpyc.utils.server import ThreadedServer

from registry import ModelSpec, find_model_by_method, load_manifest, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('inference-server')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_IN = BASE_DIR / 'rpc' / 'data' / 'in'
DATA_OUT = BASE_DIR / 'rpc' / 'data' / 'out'
PORT = 18812
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_TEXT_CHARS = 10000
MAX_SECONDS = {'medium': 380, 'sm-music': 120, 'sm-sfx': 120}
DITS = ('medium', 'sm-music', 'sm-sfx')
DECODERS = ('same-s', 'same-l')
PRECISIONS = ('fp32', 'w8a8')
TASKS = ('transcribe', 'translate')
WHISPERX_SIZES = ('tiny', 'base', 'small', 'medium', 'large-v3', 'turbo')
COMPUTE_TYPES = ('int8', 'int16', 'int8_float32', 'float32')
DEPTH_MODELS = ('small', 'base', 'large', 'metric-indoor-small', 'metric-indoor-base', 'metric-indoor-large', 'metric-outdoor-small', 'metric-outdoor-base', 'metric-outdoor-large')
REMBG_MODELS = (
    'bria-rmbg', 'birefnet-cod', 'birefnet-dis', 'birefnet-general',
    'birefnet-general-fp16', 'birefnet-general-lite', 'birefnet-hrsod',
    'birefnet-massive', 'birefnet-portrait', 'isnet-anime', 'isnet-general-use',
    'sam', 'silueta', 'u2net', 'u2net_cloth_seg', 'u2net_human_seg', 'u2netp',
)
MAX_BATCH_TEXTS = 32
MAX_LORAS = 4

JOB_SLOT = threading.Semaphore(1)


def validate_payload(data: bytes) -> bytes:
    payload = bytes(data)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f'payload exceeds {MAX_PAYLOAD_BYTES} bytes')
    return payload


def validate_text(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f'text exceeds {MAX_TEXT_CHARS} characters')
    return text


def validate_optional_text(text: str | None) -> str | None:
    if text is None:
        return None
    return validate_text(str(text))


def validate_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f'{name} must be one of {list(allowed)}')
    return value


def validate_range(value: float, low: float, high: float, name: str) -> float:
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


class InferenceService(rpyc.Service):
    def exposed_ping(self) -> str:
        return 'pong'

    def exposed_list_methods(self) -> list[str]:
        manifest = load_manifest()
        return [model.method for model in manifest.models if model.enabled]

    def exposed_text_to_speech(
        self,
        text,
        reference_audio: bytes | None = None,
        reference_transcript: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        duration: float | None = None,
        speed: float | None = None,
        normalize_text: bool = False,
        num_step: int | None = None,
        guidance_scale: float | None = None,
        t_shift: float | None = None,
        layer_penalty_factor: float | None = None,
        position_temperature: float | None = None,
        class_temperature: float | None = None,
        denoise: bool | None = None,
        audio_chunk_duration: float | None = None,
        audio_chunk_threshold: float | None = None,
        pad_duration: float | None = None,
        fade_duration: float | None = None,
        seed: int | None = None,
        voice_clone_prompt: bytes | None = None,
        preprocess_prompt: bool | None = None,
        postprocess_output: bool | None = None,
    ):
        batch = isinstance(text, list)
        if batch:
            if not 1 <= len(text) <= MAX_BATCH_TEXTS:
                raise ValueError(f'text list must contain 1..{MAX_BATCH_TEXTS} items')
            texts = [validate_text(item) for item in text]
        else:
            texts = [validate_text(text)]
        reference_transcript = validate_optional_text(reference_transcript)
        instruct = validate_optional_text(instruct)
        language = validate_optional_text(language)
        if duration is not None:
            validate_range(duration, 0.5, 600.0, 'duration')
        if speed is not None:
            validate_range(speed, 0.5, 2.0, 'speed')
        if num_step is not None:
            validate_range(num_step, 1, 200, 'num_step')
        if guidance_scale is not None:
            validate_range(guidance_scale, 0.0, 20.0, 'guidance_scale')
        if t_shift is not None:
            validate_range(t_shift, 0.0, 10.0, 't_shift')
        if layer_penalty_factor is not None:
            validate_range(layer_penalty_factor, 0.0, 50.0, 'layer_penalty_factor')
        if position_temperature is not None:
            validate_range(position_temperature, 0.01, 50.0, 'position_temperature')
        if class_temperature is not None:
            validate_range(class_temperature, 0.0, 50.0, 'class_temperature')
        if audio_chunk_duration is not None:
            validate_range(audio_chunk_duration, 1.0, 120.0, 'audio_chunk_duration')
        if audio_chunk_threshold is not None:
            validate_range(audio_chunk_threshold, 1.0, 300.0, 'audio_chunk_threshold')
        if pad_duration is not None:
            validate_range(pad_duration, 0.0, 10.0, 'pad_duration')
        if fade_duration is not None:
            validate_range(fade_duration, 0.0, 10.0, 'fade_duration')
        if voice_clone_prompt is not None and reference_audio is not None:
            raise ValueError('voice_clone_prompt and reference_audio are mutually exclusive')
        if voice_clone_prompt is not None:
            voice_clone_prompt = validate_payload(voice_clone_prompt)
        model = find_model_by_method('text_to_speech')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            arguments = ['--output', str(output_path)]
            for item in texts:
                arguments.extend(['--text', item])
            if reference_audio is not None:
                reference_path = in_dir / 'reference.wav'
                reference_path.write_bytes(validate_payload(reference_audio))
                arguments.extend(['--reference-audio', str(reference_path)])
            if reference_transcript is not None:
                arguments.extend(['--reference-transcript', reference_transcript])
            if instruct is not None:
                arguments.extend(['--instruct', instruct])
            if language is not None:
                arguments.extend(['--language', language])
            if duration is not None:
                arguments.extend(['--duration', str(duration)])
            if speed is not None:
                arguments.extend(['--speed', str(speed)])
            if normalize_text:
                arguments.append('--normalize-text')
            if num_step is not None:
                arguments.extend(['--num-step', str(num_step)])
            if guidance_scale is not None:
                arguments.extend(['--guidance-scale', str(guidance_scale)])
            if t_shift is not None:
                arguments.extend(['--t-shift', str(t_shift)])
            if layer_penalty_factor is not None:
                arguments.extend(['--layer-penalty-factor', str(layer_penalty_factor)])
            if position_temperature is not None:
                arguments.extend(['--position-temperature', str(position_temperature)])
            if class_temperature is not None:
                arguments.extend(['--class-temperature', str(class_temperature)])
            if denoise is not None:
                arguments.extend(['--denoise', str(denoise)])
            if audio_chunk_duration is not None:
                arguments.extend(['--audio-chunk-duration', str(audio_chunk_duration)])
            if audio_chunk_threshold is not None:
                arguments.extend(['--audio-chunk-threshold', str(audio_chunk_threshold)])
            if pad_duration is not None:
                arguments.extend(['--pad-duration', str(pad_duration)])
            if fade_duration is not None:
                arguments.extend(['--fade-duration', str(fade_duration)])
            if seed is not None:
                arguments.extend(['--seed', str(seed)])
            if voice_clone_prompt is not None:
                prompt_path = in_dir / 'prompt.pt'
                prompt_path.write_bytes(voice_clone_prompt)
                arguments.extend(['--voice-prompt', str(prompt_path)])
            if preprocess_prompt is not None:
                arguments.extend(['--preprocess-prompt', str(preprocess_prompt).lower()])
            if postprocess_output is not None:
                arguments.extend(['--postprocess-output', str(postprocess_output).lower()])
            return arguments

        return self._run_job(model, prepare, 'result.wav', multi=batch)

    def exposed_generate_music(
        self,
        text: str,
        seconds: float = 30.0,
        seed: int | None = None,
        dit: str = 'medium',
        dit_precision: str = 'fp32',
        decoder: str = 'same-l',
        decoder_precision: str = 'w8a8',
        encoder_precision: str = 'w8a8',
        cfg: float | None = None,
        negative_prompt: str | None = None,
        init_audio: bytes | None = None,
        init_noise_level: float | None = None,
        inpaint_range: str | None = None,
        max_rung: int | None = None,
        steps: int | None = None,
        apg: float | None = None,
        loras: list | None = None,
        lora_strength: float | None = None,
        lora_strengths: list | None = None,
        threads: int | None = None,
        free_models: bool | None = None,
    ) -> bytes:
        text = validate_text(text)
        negative_prompt = validate_optional_text(negative_prompt)
        validate_choice(dit, DITS, 'dit')
        validate_choice(dit_precision, PRECISIONS, 'dit_precision')
        validate_choice(decoder, DECODERS, 'decoder')
        validate_choice(decoder_precision, PRECISIONS, 'decoder_precision')
        validate_choice(encoder_precision, PRECISIONS, 'encoder_precision')
        validate_range(seconds, 1, MAX_SECONDS[dit], 'seconds')
        if cfg is not None:
            validate_range(cfg, 0.0, 20.0, 'cfg')
        if init_noise_level is not None:
            validate_range(init_noise_level, 0.01, 1.0, 'init_noise_level')
        if inpaint_range is not None:
            if init_audio is None:
                raise ValueError('inpaint_range requires init_audio')
            parts = str(inpaint_range).split(',')
            if len(parts) != 2:
                raise ValueError('inpaint_range must be "start,end"')
            start = float(parts[0])
            end = float(parts[1])
            if not 0 <= start < end <= seconds:
                raise ValueError('inpaint_range must satisfy 0 <= start < end <= seconds')
        if max_rung is not None and max_rung <= 0:
            raise ValueError('max_rung must be positive')
        if steps is not None:
            validate_range(steps, 1, 50, 'steps')
        if apg is not None:
            validate_range(apg, 0.0, 1.0, 'apg')
        if loras is not None:
            if not 1 <= len(loras) <= MAX_LORAS:
                raise ValueError(f'loras must contain 1..{MAX_LORAS} items')
            for lora in loras:
                validate_payload(lora)
            if dit_precision != 'fp32':
                raise ValueError('loras require dit_precision fp32')
            if lora_strengths is not None:
                if len(lora_strengths) != len(loras):
                    raise ValueError('lora_strengths must match loras length')
                for strength in lora_strengths:
                    validate_range(strength, 0.0, 10.0, 'lora_strengths')
        if lora_strength is not None:
            validate_range(lora_strength, 0.0, 10.0, 'lora_strength')
        if threads is not None:
            validate_range(threads, 1, 64, 'threads')
        model = find_model_by_method('generate_music')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            arguments = [
                '--text', text,
                '--seconds', str(seconds),
                '--dit', dit,
                '--dit-precision', dit_precision,
                '--decoder', decoder,
                '--decoder-precision', decoder_precision,
                '--encoder-precision', encoder_precision,
                '--output', str(output_path),
            ]
            if seed is not None:
                arguments.extend(['--seed', str(seed)])
            if cfg is not None:
                arguments.extend(['--cfg', str(cfg)])
            if negative_prompt is not None:
                arguments.extend(['--negative-prompt', negative_prompt])
            if init_audio is not None:
                init_path = in_dir / 'init.wav'
                init_path.write_bytes(validate_payload(init_audio))
                arguments.extend(['--init-audio', str(init_path)])
            if init_noise_level is not None:
                arguments.extend(['--init-noise-level', str(init_noise_level)])
            if inpaint_range is not None:
                arguments.extend(['--inpaint-range', str(inpaint_range)])
            if max_rung is not None:
                arguments.extend(['--max-rung', str(max_rung)])
            if steps is not None:
                arguments.extend(['--steps', str(steps)])
            if apg is not None:
                arguments.extend(['--apg', str(apg)])
            if loras is not None:
                for index, lora in enumerate(loras):
                    lora_path = in_dir / f'lora-{index}.safetensors'
                    lora_path.write_bytes(lora)
                    arguments.extend(['--lora', str(lora_path)])
                    if lora_strengths is not None:
                        arguments.extend(['--lora-strength', str(lora_strengths[index])])
                if lora_strength is not None:
                    arguments.extend(['--lora-strength-default', str(lora_strength)])
            if threads is not None:
                arguments.extend(['--threads', str(threads)])
            if free_models is not None:
                arguments.extend(['--free-models', str(free_models).lower()])
            return arguments

        return self._run_job(model, prepare, 'result.wav')

    def exposed_remove_background(
        self,
        image: bytes,
        only_mask: bool = False,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        post_process_mask: bool = False,
        bgcolor: list[int] | None = None,
        model: str = 'bria-rmbg',
        decontaminate: bool = False,
        vitmatte: bool = False,
        vitmatte_model: str | None = None,
        putalpha: bool = False,
        sam_prompt: str | None = None,
        threads: int | None = None,
    ) -> bytes:
        image = validate_payload(image)
        validate_choice(model, REMBG_MODELS, 'model')
        vitmatte_model = validate_optional_text(vitmatte_model)
        sam_prompt = validate_optional_text(sam_prompt)
        if sam_prompt is not None:
            if model != 'sam':
                raise ValueError('sam_prompt requires model=sam')
            try:
                parsed_prompt = json.loads(sam_prompt)
            except ValueError:
                raise ValueError('sam_prompt must be valid JSON') from None
            if not isinstance(parsed_prompt, list):
                raise ValueError('sam_prompt must be a JSON array')
        if threads is not None:
            validate_range(threads, 1, 64, 'threads')
        validate_range(alpha_matting_foreground_threshold, 0, 255, 'alpha_matting_foreground_threshold')
        validate_range(alpha_matting_background_threshold, 0, 255, 'alpha_matting_background_threshold')
        validate_range(alpha_matting_erode_size, 0, 255, 'alpha_matting_erode_size')
        if bgcolor is not None:
            values = [int(v) for v in bgcolor]
            if len(values) not in (3, 4) or any(not 0 <= v <= 255 for v in values):
                raise ValueError('bgcolor must be 3 or 4 ints in 0..255')
        spec = find_model_by_method('remove_background')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.png'
            input_path.write_bytes(image)
            arguments = ['--input', str(input_path), '--output', str(output_path)]
            if only_mask:
                arguments.append('--only-mask')
            if alpha_matting:
                arguments.append('--alpha-matting')
                arguments.extend(['--alpha-matting-foreground-threshold', str(alpha_matting_foreground_threshold)])
                arguments.extend(['--alpha-matting-background-threshold', str(alpha_matting_background_threshold)])
                arguments.extend(['--alpha-matting-erode-size', str(alpha_matting_erode_size)])
            if post_process_mask:
                arguments.append('--post-process-mask')
            if bgcolor is not None:
                arguments.extend(['--bgcolor', ','.join(str(int(v)) for v in bgcolor)])
            arguments.extend(['--model', model])
            if decontaminate:
                arguments.append('--decontaminate')
            if vitmatte:
                arguments.append('--vitmatte')
                if vitmatte_model is not None:
                    arguments.extend(['--vitmatte-model', vitmatte_model])
            if putalpha:
                arguments.append('--putalpha')
            if sam_prompt is not None:
                arguments.extend(['--sam-prompt', sam_prompt])
            if threads is not None:
                arguments.extend(['--threads', str(threads)])
            return arguments

        return self._run_job(spec, prepare, 'result.png')

    def exposed_interpolate_video(
        self,
        video: bytes,
        multiplier: int = 2,
        model: str = 'rife-v4.6',
        uhd: bool = False,
        tta_spatial: bool = False,
        tta_temporal: bool = False,
        jobs: str | None = None,
        gpu_id: int | None = None,
    ) -> bytes:
        video = validate_payload(video)
        validate_range(multiplier, 2, 8, 'multiplier')
        jobs = validate_optional_text(jobs)
        if jobs is not None:
            parts = jobs.split(':')
            if len(parts) != 3 or not all(part.isdigit() and int(part) > 0 for part in parts):
                raise ValueError('jobs must be "load:proc:save" positive integers')
        if gpu_id is not None and gpu_id < -1:
            raise ValueError('gpu_id must be >= -1')
        if '/' in model or '..' in model or not model.strip():
            raise ValueError('model must be a plain directory name')
        if multiplier != 2 and not model.startswith('rife-v4'):
            raise ValueError('multiplier other than 2 requires rife-v4 family model')
        spec = find_model_by_method('interpolate_video')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.mp4'
            input_path.write_bytes(video)
            arguments = [
                '--input', str(input_path),
                '--output', str(output_path),
                '--multiplier', str(multiplier),
                '--model', model,
            ]
            if uhd:
                arguments.append('--uhd')
            if tta_spatial:
                arguments.append('--tta-spatial')
            if tta_temporal:
                arguments.append('--tta-temporal')
            if jobs is not None:
                arguments.extend(['--jobs', jobs])
            if gpu_id is not None:
                arguments.extend(['--gpu-id', str(gpu_id)])
            return arguments

        return self._run_job(spec, prepare, 'result.mp4')

    def exposed_transcribe_audio(
        self,
        audio: bytes,
        language: str | None = None,
        task: str = 'transcribe',
        batch_size: int = 4,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        model_size: str = 'large-v3',
        diarize: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
        fill_nearest: bool = False,
        compute_type: str = 'int8',
        threads: int | None = None,
        beam_size: int | None = None,
        best_of: int | None = None,
        patience: float | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        log_prob_threshold: float | None = None,
        no_speech_threshold: float | None = None,
        compression_ratio_threshold: float | None = None,
        condition_on_previous_text: bool | None = None,
        prompt_reset_on_temperature: float | None = None,
        temperatures: list | None = None,
        prefix: str | None = None,
        suppress_blank: bool | None = None,
        suppress_tokens: list | None = None,
        prepend_punctuations: str | None = None,
        append_punctuations: str | None = None,
        max_new_tokens: int | None = None,
        clip_timestamps: str | None = None,
        hallucination_silence_threshold: float | None = None,
        suppress_numerals: bool | None = None,
    ) -> bytes:
        audio = validate_payload(audio)
        language = validate_optional_text(language)
        initial_prompt = validate_optional_text(initial_prompt)
        hotwords = validate_optional_text(hotwords)
        validate_choice(task, TASKS, 'task')
        validate_range(batch_size, 1, 16, 'batch_size')
        validate_choice(model_size, WHISPERX_SIZES, 'model_size')
        if min_speakers is not None:
            validate_range(min_speakers, 1, 50, 'min_speakers')
        if max_speakers is not None:
            validate_range(max_speakers, 1, 50, 'max_speakers')
        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            raise ValueError('min_speakers must not exceed max_speakers')
        if num_speakers is not None:
            validate_range(num_speakers, 1, 50, 'num_speakers')
            if min_speakers is not None or max_speakers is not None:
                raise ValueError('num_speakers is mutually exclusive with min_speakers/max_speakers')
        validate_choice(compute_type, COMPUTE_TYPES, 'compute_type')
        if threads is not None:
            validate_range(threads, 1, 64, 'threads')
        if beam_size is not None:
            validate_range(beam_size, 0, 20, 'beam_size')
        if best_of is not None:
            validate_range(best_of, 0, 20, 'best_of')
        if patience is not None:
            validate_range(patience, 0.0, 10.0, 'patience')
        if length_penalty is not None:
            validate_range(length_penalty, -10.0, 10.0, 'length_penalty')
        if repetition_penalty is not None:
            validate_range(repetition_penalty, 0.0, 10.0, 'repetition_penalty')
        if no_repeat_ngram_size is not None:
            validate_range(no_repeat_ngram_size, 0, 100, 'no_repeat_ngram_size')
        if log_prob_threshold is not None:
            validate_range(log_prob_threshold, -20.0, 0.0, 'log_prob_threshold')
        if no_speech_threshold is not None:
            validate_range(no_speech_threshold, 0.0, 1.0, 'no_speech_threshold')
        if compression_ratio_threshold is not None:
            validate_range(compression_ratio_threshold, 0.0, 10.0, 'compression_ratio_threshold')
        if prompt_reset_on_temperature is not None:
            validate_range(prompt_reset_on_temperature, 0.0, 2.0, 'prompt_reset_on_temperature')
        if temperatures is not None:
            if not 1 <= len(temperatures) <= 20:
                raise ValueError('temperatures must contain 1..20 items')
            for temperature in temperatures:
                validate_range(temperature, 0.0, 2.0, 'temperatures')
        prefix = validate_optional_text(prefix)
        if suppress_tokens is not None:
            if not 1 <= len(suppress_tokens) <= 100:
                raise ValueError('suppress_tokens must contain 1..100 items')
            for token in suppress_tokens:
                validate_range(token, -100000, 100000, 'suppress_tokens')
        prepend_punctuations = validate_optional_text(prepend_punctuations)
        append_punctuations = validate_optional_text(append_punctuations)
        if max_new_tokens is not None:
            validate_range(max_new_tokens, 1, 10000, 'max_new_tokens')
        if clip_timestamps is not None:
            clip_timestamps = validate_optional_text(clip_timestamps)
            for part in str(clip_timestamps).split(','):
                if not part:
                    raise ValueError('clip_timestamps must be comma-separated floats')
                float(part)
        if hallucination_silence_threshold is not None:
            validate_range(hallucination_silence_threshold, 0.1, 300.0, 'hallucination_silence_threshold')
        model = find_model_by_method('transcribe_audio')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.wav'
            input_path.write_bytes(audio)
            arguments = [
                '--input', str(input_path),
                '--output', str(output_path),
                '--task', task,
                '--batch-size', str(batch_size),
                '--model-size', model_size,
                '--compute-type', compute_type,
            ]
            if language is not None:
                arguments.extend(['--language', language])
            if initial_prompt is not None:
                arguments.extend(['--initial-prompt', initial_prompt])
            if hotwords is not None:
                arguments.extend(['--hotwords', hotwords])
            if diarize:
                arguments.append('--diarize')
            if min_speakers is not None:
                arguments.extend(['--min-speakers', str(min_speakers)])
            if max_speakers is not None:
                arguments.extend(['--max-speakers', str(max_speakers)])
            if num_speakers is not None:
                arguments.extend(['--num-speakers', str(num_speakers)])
            if fill_nearest:
                arguments.append('--fill-nearest')
            if threads is not None:
                arguments.extend(['--threads', str(threads)])
            if beam_size is not None:
                arguments.extend(['--beam-size', str(beam_size)])
            if best_of is not None:
                arguments.extend(['--best-of', str(best_of)])
            if patience is not None:
                arguments.extend(['--patience', str(patience)])
            if length_penalty is not None:
                arguments.extend(['--length-penalty', str(length_penalty)])
            if repetition_penalty is not None:
                arguments.extend(['--repetition-penalty', str(repetition_penalty)])
            if no_repeat_ngram_size is not None:
                arguments.extend(['--no-repeat-ngram-size', str(no_repeat_ngram_size)])
            if log_prob_threshold is not None:
                arguments.extend(['--log-prob-threshold', str(log_prob_threshold)])
            if no_speech_threshold is not None:
                arguments.extend(['--no-speech-threshold', str(no_speech_threshold)])
            if compression_ratio_threshold is not None:
                arguments.extend(['--compression-ratio-threshold', str(compression_ratio_threshold)])
            if condition_on_previous_text is not None:
                arguments.extend(['--condition-on-previous-text', str(condition_on_previous_text).lower()])
            if prompt_reset_on_temperature is not None:
                arguments.extend(['--prompt-reset-on-temperature', str(prompt_reset_on_temperature)])
            if temperatures is not None:
                arguments.extend(['--temperatures', ','.join(str(v) for v in temperatures)])
            if prefix is not None:
                arguments.extend(['--prefix', prefix])
            if suppress_blank is not None:
                arguments.extend(['--suppress-blank', str(suppress_blank).lower()])
            if suppress_tokens is not None:
                arguments.extend(['--suppress-tokens', ','.join(str(v) for v in suppress_tokens)])
            if prepend_punctuations is not None:
                arguments.extend(['--prepend-punctuations', prepend_punctuations])
            if append_punctuations is not None:
                arguments.extend(['--append-punctuations', append_punctuations])
            if max_new_tokens is not None:
                arguments.extend(['--max-new-tokens', str(max_new_tokens)])
            if clip_timestamps is not None:
                arguments.extend(['--clip-timestamps', clip_timestamps])
            if hallucination_silence_threshold is not None:
                arguments.extend(['--hallucination-silence-threshold', str(hallucination_silence_threshold)])
            if suppress_numerals is not None:
                arguments.extend(['--suppress-numerals', str(suppress_numerals).lower()])
            return arguments

        return self._run_job(model, prepare, 'result.json')

    def exposed_estimate_depth(self, image: bytes, model: str = 'small') -> bytes:
        image = validate_payload(image)
        validate_choice(model, DEPTH_MODELS, 'model')
        spec = find_model_by_method('estimate_depth')
        out_name = 'result.npy' if model.startswith('metric') else 'result.png'

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.png'
            input_path.write_bytes(image)
            return [
                '--input', str(input_path),
                '--output', str(output_path),
                '--model', model,
            ]

        return self._run_job(spec, prepare, out_name)

    def exposed_detect_scenes(self, video: bytes, threshold: float = 0.5) -> bytes:
        video = validate_payload(video)
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must be between 0 and 1 exclusive')
        model = find_model_by_method('detect_scenes')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.mp4'
            input_path.write_bytes(video)
            return ['--input', str(input_path), '--output', str(output_path), '--threshold', str(threshold)]

        return self._run_job(model, prepare, 'result.json')

    def exposed_create_voice_clone_prompt(
        self,
        reference_audio: bytes,
        reference_transcript: str | None = None,
        preprocess_prompt: bool = True,
    ) -> bytes:
        reference_audio = validate_payload(reference_audio)
        reference_transcript = validate_optional_text(reference_transcript)
        model = find_model_by_method('create_voice_clone_prompt')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            reference_path = in_dir / 'reference.wav'
            reference_path.write_bytes(reference_audio)
            arguments = [
                '--reference-audio', str(reference_path),
                '--prompt-out', str(output_path),
                '--preprocess-prompt', str(preprocess_prompt).lower(),
            ]
            if reference_transcript is not None:
                arguments.extend(['--reference-transcript', reference_transcript])
            return arguments

        return self._run_job(model, prepare, 'prompt.pt')

    def _run_job(
        self,
        model: ModelSpec,
        prepare: Callable[[Path, Path], list[str]],
        out_name: str,
        multi: bool = False,
    ):
        with JOB_SLOT:
            job_id = uuid.uuid4().hex
            in_dir = DATA_IN / job_id
            out_dir = DATA_OUT / job_id
            in_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            try:
                output_path = out_dir / out_name
                arguments = prepare(in_dir, output_path)
                cli_path = str(resolve_path(model.cli))
                if model.python:
                    command = [str(resolve_path(model.python)), cli_path, *arguments]
                else:
                    command = [cli_path, *arguments]

                logger.info('job %s started model=%s', job_id, model.name)
                started = time.monotonic()
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    _, stderr = process.communicate(timeout=model.timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
                    logger.error('job %s timed out after %ss', job_id, model.timeout_seconds)
                    raise RuntimeError(
                        f'model {model.name!r} timed out after {model.timeout_seconds}s'
                    )
                elapsed = time.monotonic() - started

                if process.returncode != 0:
                    logger.error('job %s failed: %s', job_id, stderr[-2000:])
                    raise RuntimeError(
                        f'model {model.name!r} failed (exit {process.returncode}): {stderr[-500:]}'
                    )

                logger.info('job %s done in %.1fs', job_id, elapsed)
                if multi:
                    parts = sorted(out_dir.glob(out_name.replace('.', '-*.')))
                    if not parts:
                        raise RuntimeError(f'model {model.name!r} produced no output files')
                    return [part.read_bytes() for part in parts]
                if not output_path.exists():
                    raise RuntimeError(f'model {model.name!r} produced no output file')
                return output_path.read_bytes()
            finally:
                shutil.rmtree(in_dir, ignore_errors=True)
                shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    DATA_IN.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    for stale in (*DATA_IN.iterdir(), *DATA_OUT.iterdir()):
        shutil.rmtree(stale, ignore_errors=True)
    ThreadedServer(
        InferenceService,
        port=PORT,
        protocol_config={
            'allow_all_attrs': False,
            'allow_pickle': False,
            'allow_setattr': False,
            'allow_delattr': False,
            'include_local_traceback': False,
            'include_local_version': False,
        },
    ).start()
