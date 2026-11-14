# alfa-hack-and-change

Персональные рекомендации образовательных материалов «Альфа-Курса», трек 1. Цель — подобрать предпринимателю маршрут под задачу, стадию бизнеса и опыт, объяснить выбор и следующий шаг.

Текущее состояние: каркас B01. Работают HTTP-сервис и `/api/health`; описаны общие контракты и доступна заглушка рекомендателя для разработки. Каталог, ранжирование и пользовательский интерфейс ещё не подключены. Это не сдаваемая версия продукта.

## Запуск каркаса

Нужен Docker с Compose и запущенным Linux engine:

```sh
docker compose up --build
```

Проверка: `http://localhost:8000/api/health`. Документация HTTP — `http://localhost:8000/docs`. Порт 8000 доступен только с локальной машины. Для остановки — `docker compose down`.

Health возвращает `status: scaffold`, `ready: false`, `artifacts_loaded: false`, `encoder: unavailable`, `recommender: stub`. HTTP 200 и Docker healthy означают только доступность каркаса; готовность рекомендаций ими не подтверждается. Путь `/` и продуктовые API появятся в B07.

Ключи, переменные окружения, дамп и GPU для этого этапа не нужны. Сборка загружает образ Python и пакеты; запущенный сервис внешних API не вызывает. Docker-контекст допускает только код приложения, Dockerfile и runtime-зависимости, исключая локальные данные и настройки.

## Локальная разработка

Python 3.12. Создайте окружение и установите зависимости:

```sh
python -m venv .venv
```

На Windows активируйте `.venv\Scripts\Activate.ps1`, на Linux — `source .venv/bin/activate`. Затем:

```sh
python -m pip install -r requirements-dev.txt
python -m uvicorn alfarec.api.main:app --host 127.0.0.1 --port 8000
```

Проверки:

```sh
python -m ruff format --check alfarec tests
python -m ruff check alfarec tests
python -m pytest
pwsh -NoProfile -File scripts/check.ps1
pwsh -NoProfile -File scripts/test-checks.ps1
```

Те же проверки запускаются в CI на Windows и Ubuntu с Python 3.12; сборка и запуск Docker дополнительно проверяются на Ubuntu. Детали окружения и обновления зависимостей — в [инструкции разработки](docs/development.md).

## Структура

- `alfarec/recsys/` — общие контракты, протокол `Recommender`, заглушка.
- `alfarec/api/` — FastAPI и схема запроса рекомендаций для следующего этапа.
- `alfarec/pipeline/`, `alfarec/eval/`, `alfarec/web/` — пространства пакетов для следующих задач.
- `tests/` — проверки контрактов, изоляции заглушки и HTTP-сервиса.
- `requirements.txt` и `requirements-dev.txt` — зафиксированные прямые и транзитивные зависимости.
- `requirements-ml.txt` — основа отдельного окружения ML; модели и дополнительные пакеты выбираются в B04.

Таксономия и 36 тестовых профилей проверяются командой `python -m alfarec.eval.validate_inputs`. Она проверяет конфигурацию и выводит состав набора, но не считает качество рекомендаций. [Инструкция разметки](docs/labeling.md) описывает оценки 0/1/2, разделение 22/14 и подготовку истории. Сырые связи с каталогом ещё требуют аудита дампа.

[Контракты B01](docs/specs/b01-contracts.md) описывают значения полей и ограничения заглушки. [Полное ТЗ](docs/specs/2026-11-13-recommender-tz.md) описывает целевой продукт; его возможности не следует считать реализованными.

## Работа в команде

- Вход для разработчиков и ИИ-ассистентов — [`AGENTS.md`](AGENTS.md).
- Правила разработки — [`docs/development-rules.md`](docs/development-rules.md), workflow команды — [`docs/hackathon/WORKFLOW.md`](docs/hackathon/WORKFLOW.md).
- Подготовка клона и проверки — [`docs/development.md`](docs/development.md).
- Настройка Claude Code нового участника — [`docs/claude-setup.md`](docs/claude-setup.md).
- Очередь задач — [`docs/tasks/README.md`](docs/tasks/README.md).
