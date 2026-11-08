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

    text_to_speech, create_voice_clone_prompt — контракт и параметры: models/omnivoice/README.md

    generate_music — контракт и параметры: models/stable-audio/README.md

    remove_background — контракт и параметры: models/rembg/README.md

    interpolate_video — контракт и параметры: models/rife/README.md

    transcribe_audio — контракт и параметры: models/whisperx/README.md

    estimate_depth — контракт и параметры: models/depth-anything/README.md

    detect_scenes — контракт и параметры: models/transnetv2/README.md

Общие границы: текст <= 10000 символов, payload <= 512 МБ.

Пример:

    import json
    import rpyc

    conn = rpyc.connect('<SERVER_IP>', 18812, config={'sync_request_timeout': 7200})
    wav = bytes(conn.root.text_to_speech('Hello world'))
    result = json.loads(bytes(conn.root.transcribe_audio(wav, 'en')))

sync_request_timeout — не меньше максимального timeout_seconds манифеста (7200 с); job не отменяется при разрыве соединения, слепые ретраи запрещены.

## Документация
Архитектура: [docs/architecture.md](docs/architecture.md). Контракты моделей: models/<имя>/README.md.

## Лицензия
Проект лицензирован под MIT License.
Подробнее: файл [LICENSE](LICENSE).
