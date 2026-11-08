# Depth Anything V2 (estimate_depth)

Модель: Depth Anything V2 (NeurIPS 2024, arXiv 2406.09414, DINOv2 encoder + DPT decoder). Бэкенд: transformers (AutoImageProcessor + DepthAnythingForDepthEstimation), torch 2.8.0 CPU. Все 9 чекпойнтов кэшируются и проходят smoke на установке.

## RPC-метод

    estimate_depth(image, model='small') -> bytes

## model — 9 вариантов (валидация списком на сервере)
- Относительная глубина: small (дефолт), base, large.
- Метрическая глубина indoor (NYUd): metric-indoor-small, metric-indoor-base, metric-indoor-large.
- Метрическая глубина outdoor (VKITTI): metric-outdoor-small, metric-outdoor-base, metric-outdoor-large.

## Формат выхода (зависит от семейства)
- Relative (small/base/large): PNG 16 бит, режим I;16; значения — min-max нормализованная (0..65535) относительная ИНВЕРСНАЯ глубина: меньшее значение = большая дальность (arXiv 2406.09414; upstream issue #165). Нормировка — на диапазон конкретного изображения, не метрическая.
- Metric (metric-*): NPY float32, единицы — метры. Измерено: 1073424 байта = 518 × 518 × 4 + 128 (заголовок NPY).
- Разрешение выхода обоих семейств — 518×518 (размер препроцессора), независимо от размера входного изображения; resize обратно к входному размеру не выполняется.

## Лицензии весов (README upstream, issue #320)
- Depth-Anything-V2-Small — Apache-2.0.
- Base/Large, включая метрические варианты — CC-BY-NC-4.0 (некоммерческое использование). Сервер работает в частной LAN; репозиторий (MIT) веса не распространяет — только код загрузки; скачивание весов — на машине пользователя, соблюдение CC-BY-NC-4.0 — ответственность эксплуатирующей стороны.

## Исключения (не выставляется через RPC)
- Giant (1.3B) — отсутствует в формате -hf (факт: список depth-anything/*-hf через HF API).
- prompt-depth-anything-{vits,vits-transparent,vitl}-hf — требуют входной low-res LiDAR-промпт; контракт estimate_depth(image) работает только с изображением.

## Измерения (4 CPU, 24 ГБ RAM)
small: ~5 с на job; base: ~15 с; large: ~30–40 с; metric-small: ~5–10 с. Timeout манифеста 7200 с.
