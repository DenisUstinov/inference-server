# Inference Server

Self-hosted inference-сервер, выставляющий изолированные ML-модели через единый RPC-транспорт.

## Назначение
Выделенный CPU-сервер для тяжёлых моделей ИИ: каждый RPC-вызов запускает одну модель в изолированном subprocess со своим venv; job-ы сериализуются, временные файлы удаляются после каждого job-а. Бизнес-пайплайны строятся на стороне клиента, а не внутри сервера.

## Безопасность
Только доверенная частная LAN: без аутентификации, без TLS. Машина должна оставаться внутри частной сети без пробросов портов на неё с роутера. Сервис работает под non-root пользователем; код и конфигурация принадлежат root и доступны сервису только на чтение; unit в песочнице (ProtectSystem=strict, пустой CapabilityBoundingSet, NoNewPrivileges, PrivateTmp). Вызываемы только exposed-методы; pickle, доступ к атрибутам, локальный traceback и версии отключены явно; размеры payload ограничены; все входы валидируются до записи на диск. В рантайме сервер не обращается в сеть: HF_HUB_OFFLINE=1 и TRANSFORMERS_OFFLINE=1, все веса скачиваются и кэши прогреваются на установке.

## Развёртывание

    sudo git clone https://github.com/DenisUstinov/inference-server.git /opt/inference-server
    cd /opt/inference-server
    sudo HF_TOKEN="<HF Read token>" ./bootstrap.sh

Устанавливает системные зависимости, окружения моделей, все веса (17 моделей rembg, 6 размеров whisperx, align en/ru, pipeline диаризации pyannote — лицензии должны быть приняты на HuggingFace заранее), прогревает кэши, регистрирует и запускает systemd-сервис на порту 18812. Токен (тип Read) нужен только на время установки, на диске не хранится. Установка занимает 1.5–2 часа.

## Использование
На клиентской машине:

    uv venv ~/rpc-client --python 3.12
    uv pip install --python ~/rpc-client/bin/python rpyc==6.0.2

Контракт методов (бинарные данные — bytes, структурные результаты — JSON-строки):

    ping() -> str
    list_methods() -> list[str]

    text_to_speech(text, reference_audio=None, reference_transcript=None, instruct=None,
                   language=None, duration=None, speed=None, normalize_text=False,
                   num_step=None, guidance_scale=None, t_shift=None, layer_penalty_factor=None,
                   position_temperature=None, class_temperature=None, denoise=None,
                   audio_chunk_duration=None, audio_chunk_threshold=None,
                   pad_duration=None, fade_duration=None) -> bytes | list[bytes]
        text — строка или список до 32 штук (батч: список на входе — список на выходе)
        reference_audio + reference_transcript — клонирование голоса
        instruct — дизайн голоса, контролируемый словарь: английские элементы через
        запятую с пробелом (female, male, child, teenager, young adult, middle-aged,
        elderly, whisper, high/moderate/low/very high/very low pitch, american/british/
        canadian/australian/indian/chinese/japanese/korean/portuguese/russian accent)
        либо китайские элементы через полноразмерную запятую; смешивание языков запрещено
        duration 0.5..600, speed 0.5..2.0, num_step 1..200, guidance_scale 0..20

    generate_music(text, seconds=30, seed=None, dit='medium', dit_precision='fp32',
                   decoder='same-l', decoder_precision='w8a8', encoder_precision='w8a8',
                   cfg=None, apg=None, negative_prompt=None, init_audio=None,
                   init_noise_level=None, inpaint_range=None, max_rung=None,
                   steps=None, loras=None, lora_strength=None) -> bytes
        dit: medium (seconds 1..380) | sm-music | sm-sfx (seconds 1..120)
        cfg 0..20, apg -5..5, steps 1..50, init_noise_level 0..1
        inpaint_range — строка 'start,end', 0 <= start < end <= seconds
        loras — список до 4 файлов LoRA (bytes, safetensors), lora_strength -5..5

    remove_background(image, model='bria-rmbg', only_mask=False, alpha_matting=False,
                      alpha_matting_foreground_threshold=240,
                      alpha_matting_background_threshold=10,
                      alpha_matting_erode_size=10, post_process_mask=False,
                      decontaminate=False, vitmatte=False, vitmatte_model=None,
                      bgcolor=None) -> bytes PNG
        model: bria-rmbg | birefnet-cod | birefnet-dis | birefnet-general |
        birefnet-general-fp16 | birefnet-general-lite | birefnet-hrsod |
        birefnet-massive | birefnet-portrait | isnet-anime | isnet-general-use |
        sam | silueta | u2net | u2net_cloth_seg | u2net_human_seg | u2netp
        bgcolor — 3 или 4 целых 0..255

    interpolate_video(video, multiplier=2, model='rife-v4.6', uhd=False,
                      tta_spatial=False, tta_temporal=False) -> bytes MP4
        multiplier 2..8; uhd — режим для 4K+; tta — качество ценой скорости

    transcribe_audio(audio, language=None, task='transcribe', batch_size=4,
                     initial_prompt=None, hotwords=None, model_size='large-v3',
                     diarize=False, min_speakers=None, max_speakers=None) -> bytes JSON
        task: transcribe | translate; batch_size 1..16
        model_size: tiny | base | small | medium | large-v3 | turbo
        diarize — разметка по спикерам (требует успешного align; JSON содержит
        флаги aligned/diarized и поле speaker в сегментах)
        min_speakers/max_speakers 1..50, точное число спикеров: min = max = N

    estimate_depth(image) -> bytes PNG 16 bit

    detect_scenes(video, threshold=0.5) -> bytes JSON
        threshold строго между 0 и 1

Общие границы: текст <= 10000 символов, payload <= 512 МБ.

Пример:

    import json
    import rpyc

    conn = rpyc.connect('<SERVER_IP>', 18812, config={'sync_request_timeout': 7200})
    wav = bytes(conn.root.text_to_speech('Hello world'))
    result = json.loads(bytes(conn.root.transcribe_audio(wav, 'en')))

sync_request_timeout — не меньше максимального timeout_seconds манифеста (7200 с); job не отменяется при разрыве соединения, слепые ретраи запрещены.

## Лицензия
Проект лицензирован под MIT License.
Подробнее: файл [LICENSE](LICENSE).
