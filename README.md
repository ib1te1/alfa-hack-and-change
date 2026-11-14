# alfa-hack-and-change

Персональные рекомендации образовательных материалов «Альфа-Курса», трек 1. Цель — подобрать предпринимателю маршрут под задачу, стадию бизнеса и опыт, объяснить выбор и следующий шаг.

Текущее состояние: каркас B01, каталог B02, API и страница B07. Страница и API работают на собранном каталоге. Режим `ours` пока промежуточный: порядок бейзлайна по темам, исключение пройденного и отмеченного, блок коротких материалов. ML-ранжирование, маршрут, уточнение и тексты объяснений появятся в B05 и B08. Это не сдаваемая версия продукта. Это не сдаваемая версия продукта.

## Запуск

Нужен Docker с Compose и запущенным Linux engine:

```sh
docker compose up --build
```

Страница — `http://localhost:8000/`, документация HTTP — `http://localhost:8000/docs`. Порт 8000 доступен только с локальной машины. Для остановки — `docker compose down`.

| Метод | Назначение |
|---|---|
| `GET /api/health` | `status: ok`, число материалов каталога, `recommender: interim` до подключения B05 |
| `GET /api/meta` | Стадии, уровни, задачи, сферы, форматы и бюджеты времени для онбординга |
| `GET /api/personas` | Демо-персоны из `artifacts/personas.json` с профилем и историей |
| `POST /api/recommendations` | Профиль, необязательный `user_id`, `mode` (`ours` / `baseline`), `limit` → подборки и следующий шаг |
| `POST /api/events` | Событие (`view`, `known`, `dislike` и др.); следующий запрос того же `user_id` его учитывает |
| `GET /api/items/{item_id}` | Публичная карточка материала |

События хранятся в памяти и дописываются в `data/events/events.jsonl` (путь — `EVENTS_PATH`); каталог и персоны можно переопределить через `CATALOG_PATH` и `PERSONAS_PATH`. Если каталога нет, health отвечает `status: degraded`, а рекомендации — понятной ошибкой 503.

Аккаунтов и аутентификации нет: `user_id` — анонимный идентификатор, который страница создаёт в браузере, поэтому сервис рассчитан на локальный показ, а не на публичный доступ. События демо-персон хранятся только в памяти и сбрасываются перезапуском.

Ключи, дамп и GPU для запуска не нужны; переменные окружения необязательны. Сборка загружает образ Python и пакеты; запущенный сервис внешних API не вызывает. Docker-контекст допускает только код приложения, `artifacts/`, Dockerfile и runtime-зависимости, исключая локальные данные и настройки.

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

- `alfarec/recsys/` — общие контракты, протокол `Recommender`, загрузка каталога, бейзлайн и промежуточная выдача, следующий шаг карточки.
- `alfarec/api/` — FastAPI, схемы ответов и журнал событий.
- `alfarec/web/` — страница интерфейса: HTML, CSS и JavaScript без сборки.
- `alfarec/pipeline/` — сборка каталога из дампа и отчёт о покрытии таксономии.
- `artifacts/catalog.parquet` — публичный каталог: 1497 материалов, метаданные и описание до 300 символов, без полных текстов.
- `alfarec/eval/` — тестовые профили и расчёт выборки A/B.
- `tests/` — проверки контрактов, изоляции заглушки и HTTP-сервиса.
- `requirements.txt` и `requirements-dev.txt` — зафиксированные прямые и транзитивные зависимости.
- `requirements-ml.txt` — основа отдельного окружения ML; модели и дополнительные пакеты выбираются в B04.

Таксономия и 36 тестовых профилей проверяются командой `python -m alfarec.eval.validate_inputs`. Она проверяет конфигурацию и выводит состав набора, но не считает качество рекомендаций. [Инструкция разметки](docs/labeling.md) описывает оценки 0/1/2, разделение 22/14 и подготовку истории. Связи с исходными справочниками проверены [отчётом о покрытии](reports/taxonomy_coverage.md) на собранном каталоге.

[Контракты B01](docs/specs/b01-contracts.md) описывают значения полей и ограничения заглушки. [Полное ТЗ](docs/specs/2026-11-13-recommender-tz.md) описывает целевой продукт; его возможности не следует считать реализованными.

[План A/B-проверки](docs/ab-test.md) задаёт метрики покупки и активных дней, дизайн и сроки эксперимента. `python -m alfarec.eval.sample_size` воспроизводит сценарный расчёт выборки; реальный эксперимент и аналитическая интеграция пока не запущены.

## Пересборка каталога из дампа

Дамп `alfa_course_hackathon.backup` передаётся вне репозитория и кладётся в `data/` (каталог исключён из git). PostgreSQL не нужен: дамп читается библиотекой `pgdumplib` из `requirements-dev.txt`.

```sh
python -m alfarec.pipeline.build_catalog
python -m alfarec.pipeline.coverage
python -m alfarec.pipeline.build_personas
```

Первая команда пишет `artifacts/catalog.parquet` (в git) и `data/processed/index_text.parquet` с текстом для эмбеддингов (вне git). Адрес платформы в ссылках задаёт `PLATFORM_BASE_URL` или `--base-url`; по умолчанию `https://kurs.alfabank.ru`. Вторая команда обновляет `reports/taxonomy_coverage.md`, третья — `artifacts/personas.json` по описанию `alfarec/pipeline/personas.yaml`. Тесты сверяют закоммиченные артефакты с ожидаемым составом, а при наличии дампа — побайтную воспроизводимость и отсутствие текста уроков.

## Работа в команде

- Вход для разработчиков и ИИ-ассистентов — [`AGENTS.md`](AGENTS.md).
- Правила разработки — [`docs/development-rules.md`](docs/development-rules.md), workflow команды — [`docs/hackathon/WORKFLOW.md`](docs/hackathon/WORKFLOW.md).
- Подготовка клона и проверки — [`docs/development.md`](docs/development.md).
- Настройка Claude Code нового участника — [`docs/claude-setup.md`](docs/claude-setup.md).
- Очередь задач — [`docs/tasks/README.md`](docs/tasks/README.md).
