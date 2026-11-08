# OmniVoice (text_to_speech, create_voice_clone_prompt)

Модель: k2-fsa/OmniVoice 0.2.1. Авто-транскрипция референса: openai/whisper-large-v3-turbo (кэшируется на установке). Выход: моно WAV 24000 Гц. 600+ языков, zero-shot клонирование, дизайн голоса.

## RPC-методы

    create_voice_clone_prompt(reference_audio, reference_transcript=None,
                              preprocess_prompt=True) -> bytes
        «Фотография голоса» (VoiceClonePrompt) байтами: создаётся один раз,
        переиспользуется в любых вызовах text_to_speech без повторной передачи
        референс-аудио.
        reference_transcript=None — автоматическая транскрипция референса
        (whisper-large-v3-turbo, офлайн с кэша).
        preprocess_prompt=True — удаление тишины, обрезка референса >20 с по
        наибольшей паузе, добавление пунктуации к транскрипту.
        Формат: torch.save-словарь {format_version, ref_audio_tokens, ref_text,
        ref_rms}; загрузка torch.load(weights_only=True) с проверкой версии —
        выполнение произвольного кода из байтов заблокировано (проверено:
        garbage-байты отклоняются WeightsUnpickler error).

    text_to_speech(text, reference_audio=None, reference_transcript=None, instruct=None,
                   language=None, duration=None, speed=None, normalize_text=False,
                   num_step=None, guidance_scale=None, t_shift=None, layer_penalty_factor=None,
                   position_temperature=None, class_temperature=None, denoise=None,
                   audio_chunk_duration=None, audio_chunk_threshold=None,
                   pad_duration=None, fade_duration=None, seed=None,
                   voice_clone_prompt=None, preprocess_prompt=None,
                   postprocess_output=None) -> bytes | list[bytes]

text — строка или список 1..32 (батч: список на входе — список байтов на выходе, модель грузится один раз). Текст <= 10000 символов. voice_clone_prompt и reference_audio взаимоисключающи (ValueError). В батче с клонированием референс кодируется один раз на job и переиспользуется для всех текстов.

## Режимы (трансляция model.generate)
- Обычный синтез: только text.
- Клонирование: reference_audio (WAV-байты) + reference_transcript (или voice_clone_prompt, полученный ранее из create_voice_clone_prompt).
- Дизайн голоса: instruct — контролируемый словарь. EN через запятую с пробелом: female, male, child, teenager, young adult, middle-aged, elderly, whisper, high pitch, moderate pitch, low pitch, very high pitch, very low pitch, american accent, british accent, canadian accent, australian accent, indian accent, chinese accent, japanese accent, korean accent, portuguese accent, russian accent. ZH через полноразмерную запятую: 男，女，儿童，少年，青年，中年，老年，耳语，高音调，中音调，低音调，极高音调，极低音调，东北话，云南话，四川话，宁夏话，桂林话，河南话，济南话，甘肃话，石家庄话，贵州话，陕西话，青岛话. Смешивание языков запрещено; невалидный атрибут отклоняется моделью со списком допустимых.
- Авто: без reference и instruct.

## Параметры и границы (валидация на сервере до записи на диск)
- duration 0.5..600 — фиксирует длину; приоритет над speed. Выход квантуется вверх по кадровой сетке ~0.0267 с/кадр (измерено: duration=3.0 -> 3.2 с).
- speed 0.5..2.0 — >1 быстрее и короче; игнорируется при duration.
- seed — фиксирует ГСЧ (random, numpy, torch) через omnivoice.utils.common.fix_random_seed перед загрузкой модели.
- normalize_text — WeTextProcessing (tn.chinese/tn.english); официальные TN-грамматики zh/en, прочие языки проходят без изменения (проверено на ru).
- num_step 1..200 (дефолт 32), guidance_scale 0..20 (дефолт 2.0; по upstream issue #163 может не давать слышимого эффекта), t_shift 0..10 (0.1), layer_penalty_factor 0..50 (5.0), position_temperature 0.01..50 (5.0), class_temperature 0..50 (0.0), denoise bool (True), audio_chunk_duration 1..120 (15.0), audio_chunk_threshold 1..300 (30.0; чанкование с кроссфейдом активируется при оценке длительности > threshold), pad_duration 0..10 (0.1; к обоим концам: pad=0.5 дало +38400 байт), fade_duration 0..10 (0.1; огибающая, длину не меняет).
- preprocess_prompt, postprocess_output — тристат bool|None (None = дефолт модели True). preprocess_prompt в фабрике клона: удаление тишины/обрезка/пунктуация; в generate: предобработка входного текста. postprocess_output: постобработка выхода (silence removal, RMS, фейды, паддинг; в полной мере применяется к cloning-выходу).
- language — код языка (ISO 639-3, таблица arXiv 2604.00688); None — автоопределение.

## Детерминизм (измерено)
- plain с seed: байт-идентичен между прогонами ('Parameter check', seed=42 -> 75438 байт, sha256 1e6f54f1baf4fabc... воспроизводится).
- Клон с seed: байт-идентичен внутри одного пути (два прогона reference_audio, seed=42 — sha равны).
- Кросс-путь (reference_audio vs сохранённый voice_clone_prompt, один seed): байт-идентичность НЕ гарантируется — фабрика, вызванная после fix_random_seed, смещает состояние ГСЧ. Каждый путь детерминирован отдельно.
- Без seed выход недетерминирован.

## Текст
Inline-теги [laughter], [sigh], [question-ah], [surprise-wa] и pronunciation override — свойства текста, отдельных параметров не требуют.

## Исключения (не выставляется через RPC)
- generation_config — dataclass-упаковка тех же 13 параметров генерации; все поля выставлены отдельными RPC-параметрами, включая preprocess_prompt и postprocess_output.
- voice_clone_prompt как объект не передаётся; передаётся его безопасная сериализация bytes (см. create_voice_clone_prompt).

## Зависимости
omnivoice==0.2.1, torch==2.8.0+cpu, torchaudio==2.8.0+cpu, WeTextProcessing (pynini 2.1.7), transformers 5.22.1 (ASR).

## Измерения (4 CPU, 24 ГБ RAM)
Загрузка модели ~25–30 с на job; генерация фразы ~5–10 с; создание промпта ~30–40 с (с авто-ASR ~60–90 с); timeout манифеста: text_to_speech 7200 с, create_voice_clone_prompt 1800 с.
