# Проверка интеграции B05 и B08

Дата: 15 ноября 2026 года. Проверена локальная комбинация PR #21 `f1a2939`
и исправления объяснений PR #23 `7dd25d9`. Это не слитая версия и не полная
приёмка B05/B08/B10. При наложении сохранена новая формулировка короткого
формата без длительности из #21; основной diff #23 рассчитан на `main`.

## Запуск без сети

На Windows через Docker Desktop собран образ из Dockerfile проверенной
комбинации. Образ:
`sha256:67c4e7d27a7b212928b0244402c056f304c62c699a2d6ff0ca2ee1db50a54dff`.
Контейнер запущен сразу с `--network none`; запросы выполнены внутри него
через `urllib.request`, без опубликованных портов и внешних сервисов.

- Health: `ok`, `ready=true`, `catalog_items=1497`, `encoder=tfidf`,
  `recommender=ranking`.
- Пустой профиль: 5 карточек плана, 8 стартовых, 4 коротких.
- Задача `marketing`: маршрут 3, дополнительные материалы 10, короткие 4.
- Ответы повторно валидировались моделью `RecommendationResponse`;
  текст каждой причины непустой и не длиннее 200 символов, URL действия
  совпадает с URL исходной карточки.
- После события `known` материал исключён из всех трёх блоков.

Образ собран с доступом к реестрам зависимостей; проверена работа без сети
после сборки. Это не проверка второй физической машины или всех сценариев
сдачи. Тестовый контейнер остановлен и удалён; пользовательские контейнеры
и тома не изменялись.

## Открытые дефекты ранжирования

Независимое чтение B05 и отдельное воспроизведение обнаружили три P2-дефекта
на реальном каталоге. Изменения ранкера и тестов в этом отчёте не выполнялись;
исправление остаётся в объёме исполнителя #20/#21.

| Сценарий | Ожидание | Наблюдение |
|---|---|---|
| Незавершённый подписочный курс + `banking_compliance`, `time_budget=course` | Не больше одного платного/подписочного материала в первых пяти `more` | Два подписочных: `4cbddec7-c4ce-4442-87fa-c402f26924fb` и `1fcb0c44-332a-4dce-a98c-0213a35fa5e5` |
| Незавершённый курс + `marketing`, `time_budget=course` | Одна карточка с согласованным действием | `eb371f34-34ea-4cb3-991a-cbbea147eda2` одновременно в `route` с «Начать курс» и `more[0]` с «Продолжить курс» |
| Запрос вне каталога + нейтральная оценка 3 | Уточнение сохраняется, полезного сигнала история не добавляет | Уточнение исчезает, `profile_used.tasks=[]` |

Первый дефект: `Ranker.recommend` добавляет `continuing` перед результатом
`_rerank`, который считает квоты без этой карточки. Второй: маршрут уже
сформирован до выбора продолжения, и его пересечение с продолжением не
устраняется. Третий: `_low_confidence` проверяет наличие ключей
`history_weights`, хотя нейтральная оценка создаёт нулевой вес.

Минимальное воспроизведение в окружении #21 с установленными
`requirements-retrieval.txt` (из корня соответствующей версии):

```python
from datetime import UTC, datetime

from alfarec.recsys.contracts import Event, Profile
from alfarec.recsys.ranking import load_ranker

ranker = load_ranker()

def event(item_id, kind="enroll", value=None):
    return Event(user_id="acceptance", item_id=item_id, type=kind, value=value,
                 ts=datetime(2026, 11, 15, tzinfo=UTC))

paid = ranker.recommend(
    Profile(tasks=["banking_compliance"], time_budget="course"),
    [event("4cbddec7-c4ce-4442-87fa-c402f26924fb")],
)
print([e.item.item_id for e in paid.more[:5] if e.item.access != "free"])

continued = ranker.recommend(
    Profile(tasks=["marketing"], time_budget="course"),
    [event("eb371f34-34ea-4cb3-991a-cbbea147eda2")],
)
print({e.item.item_id for e in continued.route}
      & {e.item.item_id for e in continued.more})

profile = Profile(free_text="Рассчитать траекторию космического корабля")
without = ranker.recommend(profile, [])
neutral = ranker.recommend(profile, [
    event("000ecbf2-ccb2-4305-82e2-f3ed7ce1b22c", "rate", 3.0),
])
print(without.clarification is not None, neutral.clarification is not None)
```

Наблюдаемый вывод: два ID в первом списке, один ID в пересечении, затем
`True False`. После исправления ожидаются: не больше одного ID, пустое
пересечение, `True True`. Нужны регрессионные тесты с историей, а не только
проверка тех же квот и отсутствия дублей на пустом журнале.

Ранее выявленный остаток A06 — одна причина у части карточек — описан
отдельно в [отчёте B08](https://github.com/ib1te1/alfa-hack-and-change/blob/fix/b08-evidence-based-explanations/reports/b08_explanations.md).
Ни зелёный CI, ни этот smoke-прогон не закрывают перечисленные дефекты.
