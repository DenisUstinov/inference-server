# whisperX (transcribe_audio)

Модели: whisperX 3.8.6 на faster-whisper 1.2.1 (CTranslate2, CPU); 6 размеров ASR: tiny, base, small, medium, large-v3, turbo (все закэшированы на установке); word-level alignment — wav2vec2 (en, ru); диаризация — pyannote/speaker-diarization-community-1 (лицензии HF приняты, веса в офлайн-кэше). Выход: JSON-строка (bytes).

## RPC-метод

    transcribe_audio(audio, language=None, task='transcribe', batch_size=4,
                     initial_prompt=None, hotwords=None, model_size='large-v3',
                     diarize=False, min_speakers=None, max_speakers=None,
                     num_speakers=None, fill_nearest=False, compute_type='int8',
                     threads=None, beam_size=None, best_of=None, patience=None,
                     length_penalty=None, repetition_penalty=None,
                     no_repeat_ngram_size=None, log_prob_threshold=None,
                     no_speech_threshold=None, compression_ratio_threshold=None,
                     condition_on_previous_text=None, prompt_reset_on_temperature=None,
                     temperatures=None, prefix=None, suppress_blank=None,
                     suppress_tokens=None, prepend_punctuations=None,
                     append_punctuations=None, max_new_tokens=None,
                     clip_timestamps=None, hallucination_silence_threshold=None,
                     suppress_numerals=None) -> bytes JSON

Формат JSON: {status, task, language, text, segments, word_segments, aligned, diarized, total_seconds}. word_segments — слова с start/end (и speaker при diarize=True); segments — сегменты с текстом и метками (и speaker при diarize).

## Параметры и границы (валидация на сервере до записи на диск)
- task: transcribe | translate. batch_size 1..16. language — ISO-код или None (автоопределение).
- model_size: tiny | base | small | medium | large-v3 | turbo.
- compute_type: int8 (дефолт) | int16 | int8_float32 | float32 — точный список типов, поддерживаемых CTranslate2 на CPU (измерено ctranslate2.get_supported_compute_types('cpu')). int8最快; float32 точнее (upstream фиксировал потери речи на int8 в отдельных случаях).
- threads 1..64 — cpu_threads на воркер (параметр load_model).
- initial_prompt, hotwords — контекст/лексика; проводка через asr_options на этапе load_model (whisperX issue #462: asr_options применяются при загрузке модели).
- diarize + min_speakers/max_speakers 1..50 (min <= max) либо num_speakers 1..50 (точное число; взаимоисключающе с min/max — ValueError). fill_nearest — назначение спикера ближайшим сегментом (параметр assign_word_speakers). Диаризация требует успешного align: при недоступной alignment-модели языка diarized=false без падения job-а.
- 21 опция декодирования — поля TranscriptionOptions установленной faster-whisper 1.2.1 (измерено dataclasses.fields): beam_size 0..20, best_of 0..20, patience 0..10, length_penalty -10..10, repetition_penalty 0..10, no_repeat_ngram_size 0..100, log_prob_threshold -20..0, no_speech_threshold 0..1, compression_ratio_threshold 0..10, condition_on_previous_text (bool|None), prompt_reset_on_temperature 0..2, temperatures (список 1..20 значений 0..2), prefix (строка), suppress_blank (bool|None), suppress_tokens (список 1..100 целых -100000..100000), prepend_punctuations/append_punctuations (строки), max_new_tokens 1..10000, clip_timestamps (строка comma-separated float), hallucination_silence_threshold 0.1..300, suppress_numerals (bool|None; обрабатывается whisperx до TranscriptionOptions). Диапазоны — серверные; upstream значения не валидирует.

## Исключения (не выставляется через RPC)
- without_timestamps, max_initial_timestamp, word_timestamps — внутренние тайминг-флаги конвейера whisperX: word-level таймстемпы предоставляет align, изменение флагов ломает конвейер.
- multilingual — выводится из загруженной модели.
- vad_model, vad_options, vad_method — VAD pyannote активен всегда; опции — dict-объекты.
- return_embeddings, speaker_embeddings — dict эмбеддингов дикторов, RPC-сценария нет.
- device, device_index, download_root, local_files_only, use_auth_token — инфраструктурные: CPU-фиксация, офлайн-кэш, токен только на установке.

## Известные факты
- pyannote может вернуть больше спикеров, чем max_speakers (upstream issue #677 — эвристики кластеризации); точное число — num_speakers.
- При отсутствии alignment-модели для языка: aligned=false, word_segments пустые, текст и сегменты возвращаются.
- Ошибки валидации clip_timestamps возвращают ValueError сервера до запуска job-а.

## Измерения (4 CPU, 24 ГБ RAM)
tiny: ~5–10 с на вызов; large-v3: ~30–60 с; +диаризация: ~30–60 с. Timeout манифеста 7200 с.
