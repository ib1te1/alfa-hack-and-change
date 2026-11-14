"""Next step for a material card, chosen by format and access (section 7 of the spec).

Reason texts and the route order line are added in B08.
"""

from alfarec.recsys.contracts import Item, NextStep

_FREE_STEPS = {
    "quick_answer": ("read", "Читать"),
    "guide": ("read", "Читать"),
    "template": ("download", "Скачать шаблон"),
    "tool": ("calculate", "Посчитать"),
    "course": ("start_course", "Начать курс"),
    "recording": ("watch", "Смотреть запись"),
    "podcast": ("listen", "Слушать"),
}


def next_step_for(item: Item) -> NextStep:
    if item.access == "subscription":
        kind, label = "subscribe", "Оформить Курс+"
    elif item.access == "paid":
        kind, label = "purchase", "Купить курс" if item.format == "course" else "Купить"
    else:
        kind, label = _FREE_STEPS[item.format]
    return NextStep(kind=kind, label=label, url=item.url)
