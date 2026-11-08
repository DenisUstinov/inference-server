# TransNetV2 (detect_scenes)

Модель: TransNetV2 (arXiv 2008.04838, DDCNN V2, цитирований 402), PyTorch-порт transnetv2-pytorch (pin commit e534574, project/inference-pytorch), веса MiaoshouAI/transnetv2-pytorch-weights (кэшируются на установке). Вход: видео любого контейнера, декодируемого ffmpeg. Выход: JSON-строка (bytes).

## RPC-метод

    detect_scenes(video, threshold=0.5) -> bytes JSON

Формат JSON: {status, num_frames, scenes, total_seconds}. scenes — список пар [start_frame, end_frame] включительно.

## Параметры и границы (валидация на сервере до записи на диск)
- threshold 0..1 — порог детекции: кадр считается переходом при предсказании > threshold. Семантика подтверждена измерением на тестовом видео с тремя жёсткими склейками (36 кадров): threshold 0.5 и 0.1 -> 3 сцены [[0,11],[12,23],[24,35]]; threshold 0.9 -> 1 сцена [[0,35]] (уверенность переходов ниже порога). Меньше порог — больше детектируемых переходов.

## Реализация (тонкая трансляция)
- ffmpeg: декодирование в RGB-кадры 48x27 (вход модели по публикации).
- Обработка окнами по 100 кадров (архитектурное окно), хвост дополняется нулями, padding исключается из предсказаний.
- Голова single_frame_pred + sigmoid; по публикации для предсказания переходов используется только single-frame head.
- Сцены строятся из бинарной последовательности predictions > threshold; при отсутствии переходов возвращается одна сцена на всё видео.

## Исключения (не выставляется через RPC)
- many-to-one head (all_frame_pred) — вспомогательная голова обучения; публикация: «Only the single-frame head confidence scores are utilized for transition prediction».
- Иные параметры у библиотеки отсутствуют (модель без конфигурации).

## Бенчмарки (публикация, F1)
ClipShots 77.9, BBC Planet Earth 96.2, RAI 93.9.

## Измерения (4 CPU, 24 ГБ RAM)
Видео 3 с 320x240/12fps: ~3–5 с на job. Timeout манифеста 7200 с.
