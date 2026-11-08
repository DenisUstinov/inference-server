# Stable Audio 3 (generate_music)

Модель: Stability-AI/stable-audio-3, tflite-порт, pin commit 790793b; рантайм ai-edge-litert-nightly==2.4.0.dev20261105 (xnnpack weight cache для fp32 DiT строится только nightly-версией; прогрев 14 кэшей — 12 коротких генераций на установке). Выход: стерео WAV 44100 Гц 16 бит. DiT: medium (seconds 1..380), sm-music, sm-sfx (seconds 1..120); кодеки: same-s, same-l; точности: w8a8, fp32.

## RPC-метод

    generate_music(text, seconds=30.0, seed=None, dit='medium', dit_precision='fp32',
                   decoder='same-l', decoder_precision='w8a8', encoder_precision='w8a8',
                   cfg=None, apg=None, negative_prompt=None, init_audio=None,
                   init_noise_level=None, inpaint_range=None, max_rung=None,
                   steps=None, loras=None, lora_strength=None, lora_strengths=None,
                   threads=None, free_models=None) -> bytes

Три режима (публикация arXiv 2605.17991): text-to-audio, audio-to-audio, inpainting/continuation.

## Параметры и границы (валидация на сервере до записи на диск)
- text <= 10000 символов; пустая строка = безусловная генерация (upstream: "Empty string = unconditional").
- seconds — float 1.0..380 (medium) / 1.0..120 (sm-*); T_lat = ceil(seconds*44100/4096); итоговый WAV trimmed exactly to seconds (измерено: 2.5 с -> 441044 байт).
- seed — с seed выход байт-детерминирован (эталон: 'test tone', 10 с, seed 42, medium fp32 / same-l / w8a8 -> 1764044 байт, sha256 a53e002a727803f0cdd02bd6d87aa22a407219b6e0e6d18f0c44f0495e3bb8b1); без seed upstream выбирает случайный seed и печатает его только в stdout subprocess — клиенту не возвращается.
- dit: medium | sm-music | sm-sfx; decoder: same-s | same-l; dit_precision/decoder_precision/encoder_precision: w8a8 | fp32. w8a8 — "quality-free: round-trip identical to fp32, ~3x faster, ~half RAM" (tflite README); fp32 — bit-exact референс, обязателен для LoRA.
- cfg 0..20: 1.0 = off (single pass); != 1.0 — dual pass cond+uncond. negative_prompt — текст uncond-ветки (игнорируется при cfg=1.0; при unset — all-zero T5 hidden).
- apg 0..1 — Adaptive Projected Guidance, действует только при cfg != 1.0 (1.0 = full APG, 0.0 = vanilla CFG).
- init_audio (WAV-байты) + init_noise_level 0.01..1.0 — audio-to-audio; init_noise_level = sigma_max: 0.4-0.8 вариации, 1.0 полная перегенерация; нижняя граница MIN_SIGMA=0.01 (ниже — NaN).
- inpaint_range 'start,end' — требует init_audio; 0 <= start < end <= seconds; регенерирует диапазон, сохраняет остальное (paste-back).
- max_rung > 0 — потолок rung-уровня SAME-кодека (снижает пиковый RAM).
- steps 1..50 — шаги pingpong-сэмплера (дефолт 8; rf_denoiser дистиллирован под 8).
- loras — список 1..4 файлов .safetensors (SA3-native train_lora.py / underfit или PEFT; принимается только .safetensors); требуют dit_precision='fp32' (int8-DiT не сливается); слияние при загрузке DiT, кэш models/tflite/lora_cache/ — единственный каталог записи в дереве кода (ReadWritePaths юнита + chown сервисному пользователю на установке).
- lora_strengths — список сил 0..10 по адаптерам (длина = длине loras; 0 отключает — bit-identical без LoRA; >1 усиливает).
- lora_strength 0..10 — глобальный дефолт силы для адаптеров без индивидуальной.
- threads 1..64 — потоки XNNPACK CPU (дефолт upstream 8).
- free_models — bool|None: освобождение моделей после последнего использования против пикового RAM (дефолт upstream True; False — резидентность в пределах job).

## Исключения (не выставляется через RPC)
- precision — глобальный алиас (DiT + маппинг codec fp32->fp32, int8->w8a8); полностью покрыт dit_precision/decoder_precision/encoder_precision.
- cfg-batched — no-op для rung-DiT в pinned-сборке (tflite README); батч-режим относится к variable-batch DiT.
- play — локальное воспроизведение на сервере (afplay/aplay), за RPC бессмысленно.
- out — внутренний выходной файл job-а.

## Известные факты
- Невалидные LoRA-байты: upstream-парсер safetensors трактует мусорный заголовок как длину и падает с MemoryError (lora_core._load_safetensors) — job завершается RuntimeError, сервис продолжает работу (проверено).
- Позитивный тест слияния LoRA требует валидный .safetensors-адаптер SA3; проверены: требование fp32, per-adapter трансляция (--lora path [strength=S]), доступность кэша на запись.
- Первый запуск новой комбинации dit/precision до построения xnnpack-кэша дольше; кэш строится однократно.

## Измерения (4 CPU, 24 ГБ RAM)
medium fp32 / same-l / w8a8, 10 с: ~90 с (тёплый кэш); sm-music w8a8 / same-s, 3 с: ~20 с; timeout манифеста 7200 с.
