# rembg (remove_background)

Модель: rembg 2.0.86, ONNX Runtime (CPU). 17 сегментационных моделей, все веса скачиваются и прогреваются на установке. Выход: PNG.

## RPC-метод

    remove_background(image, only_mask=False, alpha_matting=False,
                      alpha_matting_foreground_threshold=240,
                      alpha_matting_background_threshold=10,
                      alpha_matting_erode_size=10, post_process_mask=False,
                      bgcolor=None, model='bria-rmbg', decontaminate=False,
                      vitmatte=False, vitmatte_model=None, putalpha=False,
                      sam_prompt=None, threads=None) -> bytes PNG

## model — выбор сегментационной модели
bria-rmbg (дефолт, 2.0), birefnet-cod, birefnet-dis, birefnet-general, birefnet-general-fp16, birefnet-general-lite, birefnet-hrsod, birefnet-massive, birefnet-portrait, isnet-anime, isnet-general-use, sam, silueta, u2net, u2net_cloth_seg, u2net_human_seg, u2netp. Невалидное имя — ValueError со списком. Семейство BiRefNet: варианты General/Portrait/HR (HRSOD)/COD/Massive обучены на разных задачах выделения; birefnet-general-fp16 конвертируется из fp32 при первой установке (предупреждения onnxconverter_common о truncation до 1e-07 — штатный шум конвертации).

## Параметры и границы (валидация на сервере до записи на диск)
- only_mask — вернуть только маску.
- alpha_matting + alpha_matting_foreground_threshold (0..255, дефолт 240), alpha_matting_background_threshold (0..255, дефолт 10), alpha_matting_erode_size (0..255, дефолт 10) — уточнение краёв постобработкой.
- post_process_mask — морфологическая постобработка маски.
- bgcolor — список из 3 или 4 целых 0..255 (фон композитинга, RGBA).
- decontaminate — устранение цветовой каймы на мягких краях. Факт из исходника bg.py: игнорируется при alpha_matting=True.
- vitmatte + vitmatte_model — уточнение альфы нейросетью ViTMatte; имеет приоритет над alpha_matting; варианты vitmatte_model: small-distinctions-646 (дефолт), small-composition-1k, base-distinctions-646, base-composition-1k (все закэшированы на установке).
- putalpha — True: результат с применённым альфа-каналом (putalpha_cutout); False (дефолт): decontaminate-композитинг. Измерено: 5049 vs 5265 байт на тестовом изображении.
- sam_prompt — только model='sam', иначе ValueError. JSON-строка, массив элементов вида {"type": "point", "label": 0|1, "data": [x, y]}; label 1 = передний план, 0 = фон. Дефолт модели (без sam_prompt) — одна центральная точка с label 1; промпт, равный дефолтному, даёт байт-идентичный результат (проверено). Валидация сервера: валидный JSON + массив; схема элементов проверяется jsonschema внутри модели (нарушение схемы — RuntimeError).
- threads 1..64 — число потоков ONNX Runtime через OMP_NUM_THREADS (inter/intra op threads); на результат не влияет (проверено), влияет на скорость.

## Исключения (не выставляется через RPC)
- force_return_bytes — внутренний флаг формата возврата (cli всегда сохраняет PNG через PIL).
- session (объект BaseSession) — не примитив; эквивалент — параметр model.
- providers / sess_opts (объекты ONNX Runtime) — сервер CPU-only, провайдер фиксирован; потоки выставлены примитивом threads.
- Сессии ben_custom, dis_custom, u2net_custom (требуют пользовательские файлы моделей), atlascloud, withoutbg (внешние облачные сервисы — несовместимы с офлайн-рантаймом).

## Измерения (4 CPU, 24 ГБ)
bria-rmbg/u2net/isnet/silueta: секунды на изображение 256x256; birefnet-massive и sam — до минут на больших изображениях. Timeout манифеста 1800 с.
