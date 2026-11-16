"""Card explanations and the next step (section 7 of the spec).

Texts are built only from reason codes, their params and the taxonomy: no promises of results
and no facts that are not in the data. Params may carry taxonomy ids or ready labels.

Params by code:
- task_match: task; industry_match: industry; stage_fit: stage;
- continues_history: title (the material from the history);
- short_format: minutes (optional: a calculator has no recorded time);
- material_format: format (a fact about the material, added here when fewer than two other
  reasons can be worded; ranking never produces it and it carries no weight).
Popularity is aggregate, without a user cohort.
"""

from collections.abc import Callable, Mapping

from alfarec.recsys.contracts import (
    Item,
    NextStep,
    Reason,
    ReasonCode,
    Recommendation,
    RecommendationResponse,
)
from alfarec.recsys.taxonomy import Taxonomy

MAX_REASONS = 3
MAX_REASON_LENGTH = 200
MIN_REASONS = 2

_FORMAT_FACTS = {
    "quick_answer": "Короткий ответ по теме",
    "guide": "Подробный разбор темы",
    "template": "Готовый шаблон: скачайте и заполните",
    "tool": "Калькулятор: результат сразу",
    "course": "Курс по теме",
    "recording": "Запись вебинара: смотрите в удобное время",
    "podcast": "Выпуск подкаста: можно слушать в дороге",
}

_FREE_STEPS = {
    "quick_answer": ("read", "Читать"),
    "guide": ("read", "Читать"),
    "template": ("download", "Скачать шаблон"),
    "tool": ("calculate", "Посчитать"),
    "course": ("start_course", "Начать курс"),
    "recording": ("watch", "Смотреть запись"),
    "podcast": ("listen", "Слушать"),
}

# Route roles from section 6.3: understand, do, go deeper.
_ROLE_BY_FORMAT = {
    "quick_answer": "understand",
    "guide": "understand",
    "template": "do",
    "tool": "do",
    "course": "deepen",
    "recording": "deepen",
    "podcast": "deepen",
}
_ROLE_TEXT = {
    "understand": "разберёмся в основах",
    "do": "перейдём к шаблонам и расчётам",
    "deepen": "закрепим системным материалом",
}
_ORDER_WORDS = ("Сначала", "затем", "после —")


def next_step_for(item: Item) -> NextStep:
    if item.access == "subscription":
        kind, label = "subscribe", "Оформить Курс+"
    elif item.access == "paid":
        kind, label = "purchase", "Купить курс" if item.format == "course" else "Купить"
    else:
        kind, label = _FREE_STEPS[item.format]
    return NextStep(kind=kind, label=label, url=item.url)


def _label(value: object, vocabulary: Mapping) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    entry = vocabulary.get(value)
    return (entry.label if entry else value).strip()


def _lower_first(text: str) -> str:
    # Keep abbreviations such as "ИИ" or "ВЭД" intact.
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _with_label(template: str, label: str) -> str:
    # Reserve room for the complete sentence and closing quote, not just the source title.
    available = MAX_REASON_LENGTH - len(template.format(""))
    return template.format(_shorten(label, available))


def _minutes(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 1:
        return None
    return round(value)


def reason_text(reason: Reason, taxonomy: Taxonomy) -> str | None:
    """Human wording of one reason, or None when its params are not enough to be specific."""
    params = reason.params
    builders: dict[ReasonCode, Callable[[], str | None]] = {
        ReasonCode.TASK_MATCH: lambda: (
            # "about": the material's own task, used when the user named none that matches.
            _with_label(
                "Материал по теме «{}»" if params.get("about") else "Подходит под вашу задачу «{}»",
                _lower_first(task),
            )
            if (task := _label(params.get("task"), taxonomy.tasks))
            else "Подходит под вашу задачу"
        ),
        ReasonCode.INDUSTRY_MATCH: lambda: (
            _with_label("Для сферы «{}»", industry)
            if (industry := _label(params.get("industry"), taxonomy.industries))
            else "Для вашей сферы"
        ),
        ReasonCode.STAGE_FIT: lambda: (
            _with_label("Подходит для этапа «{}»", _lower_first(stage))
            if (stage := _label(params.get("stage"), taxonomy.stages))
            else "Подходит для вашего этапа"
        ),
        ReasonCode.BEGINNER_FRIENDLY: lambda: "Для тех, кто только начинает",
        ReasonCode.CONTINUES_HISTORY: lambda: (
            _with_label("По теме «{}» из вашей истории", title)
            if (title := _label(params.get("title"), {}))
            else "По теме материалов из вашей истории"
        ),
        ReasonCode.POPULAR_WITH_SIMILAR: lambda: "Популярный материал на платформе",
        ReasonCode.SHORT_FORMAT: lambda: (
            f"Займёт {minutes} мин"
            if (minutes := _minutes(params.get("minutes")))
            else "Готовый шаблон: скачайте и заполните"
            if params.get("format") == "template"
            else "Короткий формат: ответ сразу"
        ),
        ReasonCode.FRESH: lambda: "Материал об изменениях в законах",
        ReasonCode.NEXT_LEVEL: lambda: "Следующий шаг после того, что вы уже знаете",
        ReasonCode.DEEP_DIVE_COURSE: lambda: "Для более глубокого изучения темы",
        ReasonCode.MATERIAL_FORMAT: lambda: _FORMAT_FACTS.get(str(params.get("format"))),
    }
    text = builders[reason.code]()
    return _shorten(text, MAX_REASON_LENGTH) if text else None


def explain_recommendation(recommendation: Recommendation, taxonomy: Taxonomy) -> Recommendation:
    """Keep the strongest reasons that can be worded and attach their texts.

    A positive history contribution always keeps a place: it is the one reason personal to the
    user, and with small history weights it would otherwise never reach the top three. A card
    with fewer than two worded reasons gets a fact about its format. Neither rule touches the
    order of cards or their scores.
    """
    worded: list[Reason] = []
    seen: set[str] = set()
    for reason in sorted(recommendation.reasons, key=lambda reason: -reason.weight):
        if reason.weight < 0:
            continue
        text = reason_text(reason, taxonomy)
        if text and text not in seen:
            seen.add(text)
            worded.append(reason.model_copy(update={"text": text}))
    explained = worded[:MAX_REASONS]
    history = next(
        (r for r in worded if r.code == ReasonCode.CONTINUES_HISTORY and r.weight > 0), None
    )
    if history is not None and history not in explained:
        explained[-1] = history
    if len(explained) < MIN_REASONS:
        fact = Reason(
            code=ReasonCode.MATERIAL_FORMAT,
            weight=0.0,
            params={"format": recommendation.item.format},
        )
        text = reason_text(fact, taxonomy)
        if text and text not in seen:
            explained.append(fact.model_copy(update={"text": text}))
    return recommendation.model_copy(update={"reasons": explained})


def route_note(route: list[Recommendation]) -> str | None:
    """One line on the order of the route, from the roles of its steps."""
    roles: list[str] = []
    for step in route:
        role = _ROLE_BY_FORMAT[step.item.format]
        if role not in roles:
            roles.append(role)
    if len(roles) < 2:
        return None
    parts = [f"{word} {_ROLE_TEXT[role]}" for word, role in zip(_ORDER_WORDS, roles, strict=False)]
    return ", ".join(parts) + "."


def explain_response(
    response: RecommendationResponse, taxonomy: Taxonomy
) -> RecommendationResponse:
    def explain(items: list[Recommendation]) -> list[Recommendation]:
        return [explain_recommendation(item, taxonomy) for item in items]

    return response.model_copy(
        update={
            "route": explain(response.route),
            "more": explain(response.more),
            "quick": explain(response.quick),
            "route_note": route_note(response.route),
        }
    )
