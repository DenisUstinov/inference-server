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
            return arguments

        outputs = self._run_job(model, prepare, 'result.wav', multi=batch)
        return outputs if batch else outputs[0]

    def exposed_generate_music(
        self,
        text: str,
        seconds: int = 30,
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
            validate_range(init_noise_level, 0.0, 1.0, 'init_noise_level')
        if inpaint_range is not None:
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
            validate_range(apg, -5.0, 5.0, 'apg')
        if loras is not None:
            if not 1 <= len(loras) <= MAX_LORAS:
                raise ValueError(f'loras must contain 1..{MAX_LORAS} items')
            for lora in loras:
                validate_payload(lora)
        if lora_strength is not None:
            validate_range(lora_strength, -5.0, 5.0, 'lora_strength')
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
                lora_paths = []
                for index, lora in enumerate(loras):
                    lora_path = in_dir / f'lora-{index}.safetensors'
                    lora_path.write_bytes(lora)
                    lora_paths.append(str(lora_path))
                arguments.append('--lora')
                arguments.extend(lora_paths)
                if lora_strength is not None:
                    arguments.extend(['--lora-strength', str(lora_strength)])
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
    ) -> bytes:
        image = validate_payload(image)
        validate_choice(model, REMBG_MODELS, 'model')
        vitmatte_model = validate_optional_text(vitmatte_model)
        validate_range(alpha_matting_foreground_threshold, 0, 255, 'alpha_matting_foreground_threshold')
        validate_range(alpha_matting_background_threshold, 0, 255, 'alpha_matting_background_threshold')
        validate_range(alpha_matting_erode_size, 0, 255, 'alpha_matting_erode_size')
        if bgcolor is not None:
            values = [int(v) for v in bgcolor]
            if len(values) not in (3, 4) or any(not 0 <= v <= 255 for v in values):
                raise ValueError('bgcolor must be 3 or 4 ints in 0..255')
        model = find_model_by_method('remove_background')

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
            return arguments

        return self._run_job(model, prepare, 'result.png')

    def exposed_interpolate_video(
        self,
        video: bytes,
        multiplier: int = 2,
        model: str = 'rife-v4.6',
        uhd: bool = False,
        tta_spatial: bool = False,
        tta_temporal: bool = False,
    ) -> bytes:
        video = validate_payload(video)
        validate_range(multiplier, 2, 8, 'multiplier')
        if '/' in model or '..' in model or not model.strip():
            raise ValueError('model must be a plain directory name')
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
            return arguments

        return self._run_job(model, prepare, 'result.json')

    def exposed_estimate_depth(self, image: bytes) -> bytes:
        image = validate_payload(image)
        model = find_model_by_method('estimate_depth')

        def prepare(in_dir: Path, output_path: Path) -> list[str]:
            input_path = in_dir / 'input.png'
            input_path.write_bytes(image)
            return ['--input', str(input_path), '--output', str(output_path)]

        return self._run_job(model, prepare, 'result.png')

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
