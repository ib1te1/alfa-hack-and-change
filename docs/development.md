# Подготовка клона и проверки

Нужны Git с Git Credential Manager, PowerShell 7 (`pwsh`) и Python 3.12. Для контейнера — Docker с Compose и Linux engine. Проверки документов репозитория не устанавливают пакеты, не вызывают внешние сервисы и не используют платные API.

## Подготовить свой клон

1. Задать локальную Git-личность своего аккаунта GitHub, не трогая глобальную:

   ```powershell
   git config --local user.name <логин>
   git config --local user.email <ID>+<логин>@users.noreply.github.com
   git config --show-origin --get user.email
   ```

   ID аккаунта виден в `https://api.github.com/users/<логин>`. Последняя команда должна показать `.git/config`.

2. Включить защиту `main` и исключить локальные заметки:

   ```powershell
   pwsh -NoProfile -File scripts/setup-hooks.ps1
   git config --local --get core.hooksPath
   ```

   Ожидается `.githooks`. Команда включает `.githooks` только для текущего клона и исключает `.work/` через `.git/info/exclude`. При уже установленном другом наборе hooks нужно разрешить конфликт, а не отключать существующую защиту.

Авторизация (кто пушит) и автор коммита проверяются отдельно; секреты не передаются между участниками. Удалённые операции идут через Git Credential Manager: при первом обращении он откроет вход в браузере.

## Проверить изменения перед коммитом и PR

```powershell
pwsh -NoProfile -File scripts/check.ps1
pwsh -NoProfile -File scripts/test-checks.ps1
```

[Проверка репозитория](../scripts/check.ps1) проверяет Markdown, локальные ссылки, а после появления карточек и требований — граф задач и покрытие требований (формат — в [очереди задач](tasks/README.md)). Внешние ссылки не скачиваются.

[Проверочные сценарии](../scripts/test-checks.ps1) проверяют, что реальные дефекты обнаруживаются, а защита `main` отклоняет прямой, принудительный и удаляющий push. Защита проверяется на временных локальных репозиториях; тесты не пушат в рабочую `main` и не используют сеть.

В [CI](../.github/workflows/ci.yml) запускаются те же команды на Ubuntu и Windows — на каждом PR в `main` и после изменения `main`. Дополнительно выполняются Ruff и pytest на Python 3.12; на Ubuntu — сборка и запуск Docker. Зелёный CI означает прохождение перечисленных проверок, а не готовность продукта.

## Проверки Python и зависимости

Создание окружения и запуск описаны в [README](../README.md). После установки `requirements-dev.txt`:

```sh
python -m ruff format --check alfarec tests
python -m ruff check alfarec tests
python -m pytest
```

Прямые зависимости фиксируются в `requirements.in` и `requirements-dev.in`, полные наборы — в `requirements.txt` и `requirements-dev.txt`. При изменении зависимостей пересобрать оба файла с помощью `uv` и проверить на Windows и Linux:

```sh
uv pip compile requirements.in --universal --no-header --no-annotate --output-file requirements.txt
uv pip compile requirements-dev.in --universal --no-header --no-annotate --output-file requirements-dev.txt
```

`uv` нужен только для обновления списков; установка и запуск используют обычный pip. `requirements-ml.txt` пока включает runtime-набор; зависимости обучения будут добавлены после выбора моделей в B04. В Docker они не устанавливаются.

## Защита основной ветки и авторство

Локальный `pre-push` запрещает изменения и удаление удалённой `refs/heads/main`, включая push из другой локальной ветки. Это клиентская защита: каждый клон устанавливает её сам. Обход через `--no-verify` и отключение hooks запрещены правилами проекта.

Репозиторий приватный; серверная защита веток для приватных репозиториев на бесплатном тарифе GitHub недоступна. Поэтому до каждого merge отдельно проверяются зелёный CI на текущем HEAD, актуальность `main`, замечания ревью и явное разрешение на конкретный squash merge.

В настройках репозитория разрешён только squash, автоматическое слияние отключено, удаление рабочей ветки после merge включено. Это не заменяет разрешения пользователя на merge.

Перед публикацией читать сообщения коммитов и окончательное описание PR.
Общий порядок работы — в [правилах разработки](development-rules.md).
