"""Report how taxonomy source links cover the built catalog.

Usage: python -m alfarec.pipeline.coverage [--catalog PATH] [--output PATH]
"""

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import Item
from alfarec.recsys.taxonomy import Taxonomy, load_taxonomy, match_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "taxonomy_coverage.md"


# Each source link kind is compared with one catalog field.
def _article_subject(item: Item) -> list[str]:
    return [item.subject] if item.item_type == "article" and item.subject else []


SOURCES: dict[str, Callable[[Item], list[str]]] = {
    "topics": lambda item: item.topics,
    "tags": lambda item: item.raw_tags,
    "article_subjects": _article_subject,
}
SOURCE_TITLES = {"topics": "темам", "tags": "тегам", "article_subjects": "рубрикам статей"}


@dataclass
class Coverage:
    items: int
    task_items: dict[str, dict[str, int]]
    task_total: dict[str, int]
    industry_items: dict[str, int]
    missing: list[str]
    unlinked: dict[str, Counter[str]]
    without_task: Counter[str]


def _keys(values: Sequence[str]) -> set[str]:
    return {match_key(value) for value in values}


def compute_coverage(taxonomy: Taxonomy, catalog: Sequence[Item]) -> Coverage:
    present = {
        kind: _keys([v for item in catalog for v in get(item)]) for kind, get in SOURCES.items()
    }
    present["industries"] = _keys([v for item in catalog for v in item.raw_industries])
    missing: list[str] = []
    linked: dict[str, set[str]] = {kind: set() for kind in [*SOURCES, "industries"]}
    task_items: dict[str, dict[str, int]] = {}
    task_total: dict[str, int] = {}
    matched_any: set[str] = set()
    for task_id, task in taxonomy.tasks.items():
        task_items[task_id] = {}
        matched: set[str] = set()
        for kind, get in SOURCES.items():
            wanted = _keys(getattr(task.source_links, kind))
            linked[kind] |= wanted
            for value in getattr(task.source_links, kind):
                if match_key(value) not in present[kind]:
                    missing.append(f"tasks.{task_id}.{kind}: {value}")
            hits = {item.item_id for item in catalog if _keys(get(item)) & wanted}
            task_items[task_id][kind] = len(hits)
            matched |= hits
        task_total[task_id] = len(matched)
        matched_any |= matched
    industry_items: dict[str, int] = {}
    for industry_id, industry in taxonomy.industries.items():
        wanted = _keys(industry.raw_values)
        linked["industries"] |= wanted
        # A sphere without materials is still a valid profile answer: it ranks as "for everyone".
        industry_items[industry_id] = sum(
            1 for item in catalog if _keys(item.raw_industries) & wanted
        )
    getters = {**SOURCES, "industries": lambda item: item.raw_industries}
    unlinked = {
        kind: Counter(v for item in catalog for v in get(item) if match_key(v) not in linked[kind])
        for kind, get in getters.items()
    }
    without_task = Counter(item.item_type for item in catalog if item.item_id not in matched_any)
    return Coverage(
        len(catalog), task_items, task_total, industry_items, missing, unlinked, without_task
    )


def render_report(taxonomy: Taxonomy, coverage: Coverage) -> str:
    lines = [
        "# Покрытие таксономии на каталоге",
        "",
        "Отчёт создаётся командой `python -m alfarec.pipeline.coverage`"
        " по `artifacts/catalog.parquet` и `alfarec/recsys/taxonomy.yaml`."
        " Значения сравниваются по нормализованному ключу:"
        " регистр, «ё», латинские буквы-двойники.",
        "",
        f"Материалов в каталоге: {coverage.items}. Связь через исходные справочники — слабый сигнал"
        " для генерации кандидатов, а не готовая метка задачи: широкие темы относятся к нескольким"
        " задачам.",
        "",
        "## Ненайденные связи",
        "",
    ]
    if coverage.missing:
        lines += [f"- {entry}" for entry in coverage.missing]
    else:
        lines.append("Нет: каждое значение `source_links` задач встречается в каталоге.")
    lines += [
        "",
        "## Задачи",
        "",
        "| Задача | Материалов | По темам | По тегам | По рубрикам статей |",
        "|---|---|---|---|---|",
    ]
    for task_id, task in taxonomy.tasks.items():
        counts = coverage.task_items[task_id]
        lines.append(
            f"| {task.label} (`{task_id}`) | {coverage.task_total[task_id]} | {counts['topics']}"
            f" | {counts['tags']} | {counts['article_subjects']} |"
        )
    without = ", ".join(f"{kind} {count}" for kind, count in sorted(coverage.without_task.items()))
    lines += [
        "",
        f"Без связи ни с одной задачей: {sum(coverage.without_task.values())} ({without}). Для них"
        " задача восстанавливается из текста в B04.",
        "",
        "## Сферы",
        "",
        "| Сфера | Материалов |",
        "|---|---|",
    ]
    for industry_id, industry in taxonomy.industries.items():
        lines.append(
            f"| {industry.label} (`{industry_id}`) | {coverage.industry_items[industry_id]} |"
        )
    empty = [
        industry.label
        for key, industry in taxonomy.industries.items()
        if not coverage.industry_items[key]
    ]
    if empty:
        lines += [
            "",
            f"Без материалов: {', '.join(empty)}. Значение есть в справочнике `Industries`, но у"
            " опубликованных материалов не проставлено: отраслевого сигнала для такого профиля в"
            " исходной разметке нет.",
        ]
    lines += ["", "## Значения каталога без связи с таксономией", ""]
    titles = {**SOURCE_TITLES, "industries": "отраслям"}
    for kind, counter in coverage.unlinked.items():
        values = ", ".join(f"{value} ({count})" for value, count in sorted(counter.items()))
        lines.append(f"- По {titles[kind]}: {values or 'нет'}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    taxonomy = load_taxonomy()
    coverage = compute_coverage(taxonomy, load_catalog(args.catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(taxonomy, coverage), encoding="utf-8", newline="\n")
    print(f"Report: {args.output}; missing links: {len(coverage.missing)}")
    return 1 if coverage.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
