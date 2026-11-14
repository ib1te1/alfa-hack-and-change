"""Card explanations and the next step (section 7 of the spec).

Texts are built only from reason codes, their params and the taxonomy: no promises of results
and no facts that are not in the data. Params may carry taxonomy ids or ready labels.

Params by code:
- task_match: task; industry_match: industry; stage_fit: stage;
- continues_history: title (the material from the history);
- short_format: minutes; popular_with_similar: count (optional).
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


def _minutes(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 1:
        return None
    return round(value)


def reason_text(reason: Reason, taxonomy: Taxonomy) -> str | None:
    """Human wording of one reason, or None when its params are not enough to be specific."""
    params = reason.params
    builders: dict[ReasonCode, Callable[[], str | None]] = {
        ReasonCode.TASK_MATCH: lambda: (
            f"Подходит под вашу задачу «{_lower_first(task)}»"
            if (task := _label(params.get("task"), taxonomy.tasks))
            else "Подходит под вашу задачу"
        ),
        ReasonCode.INDUSTRY_MATCH: lambda: (
            f"Для сферы «{industry}»"
            if (industry := _label(params.get("industry"), taxonomy.industries))
            else "Для вашей сферы"
        ),
        ReasonCode.STAGE_FIT: lambda: (
            f"Подходит для этапа «{_lower_first(stage)}»"
            if (stage := _label(params.get("stage"), taxonomy.stages))
            else "Подходит для вашего этапа"
        ),
        ReasonCode.BEGINNER_FRIENDLY: lambda: "Для тех, кто только начинает",
        ReasonCode.CONTINUES_HISTORY: lambda: (
            f"Продолжает «{title}», который вы уже изучали"
            if (title := _label(params.get("title"), {}))
            else "Продолжает то, что вы уже изучали"
        ),
        ReasonCode.POPULAR_WITH_SIMILAR: lambda: "Популярно у предпринимателей с похожими задачами",
        ReasonCode.SHORT_FORMAT: lambda: (
            f"Займёт {minutes} мин" if (minutes := _minutes(params.get("minutes"))) else None
        ),
        ReasonCode.FRESH: lambda: "Свежий материал об изменениях",
        ReasonCode.NEXT_LEVEL: lambda: "Следующий шаг после того, что вы уже знаете",
        ReasonCode.DEEP_DIVE_COURSE: lambda: "Системный курс, чтобы разобраться глубже",
    }
    return builders[reason.code]()


def explain_recommendation(recommendation: Recommendation, taxonomy: Taxonomy) -> Recommendation:
    """Keep the strongest reasons that can be worded and attach their texts."""
    ranked = sorted(recommendation.reasons, key=lambda reason: -reason.weight)
    explained: list[Reason] = []
    seen: set[str] = set()
    for reason in ranked:
        text = reason_text(reason, taxonomy)
        if text and text not in seen:
            seen.add(text)
            explained.append(reason.model_copy(update={"text": text}))
        if len(explained) == MAX_REASONS:
            break
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
