"""Offline evaluation, stage 1 of B06 (spec section 9): checks that need no human labels.

Usage: python -m alfarec.eval [--handoff DIR] [--output PATH]

1. Next course: users with two or more courses; the latest course is hidden and predicted.
2. Editorial links: from a source material, retrieve its "see also" items.
3. Profile lists: answers for the 36 test profiles, ours against the baseline.
Relevance judgments, ablations and weight tuning are later stages and are not reported here.
"""

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from alfarec.eval.profiles import load_profiles
from alfarec.recsys.contracts import Event, Profile
from alfarec.recsys.ranking import Ranker, load_ranker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF = ROOT / "data" / "handoff"
DEFAULT_REPORT = ROOT / "reports" / "eval.md"
K = 10


def hit_rate(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def recall(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    gain = sum(
        1 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def next_course_cases(rows: Sequence[dict]) -> list[tuple[list[dict], str]]:
    """(history rows, hidden course) for users whose latest course is unambiguous."""
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["user_key"]].append(row)
    cases = []
    for user in sorted(by_user):
        entries = sorted(by_user[user], key=lambda r: (r["enrolled_at"], r["course_id"]))
        if len({r["course_id"] for r in entries}) < 2:
            continue
        last = entries[-1]
        # Dates are precise to a month: a tie for the latest course has no true answer.
        if sum(r["enrolled_at"] == last["enrolled_at"] for r in entries) > 1:
            continue
        cases.append((entries[:-1], last["course_id"]))
    return cases


def evaluate_next_course(ranker: Ranker, rows: Sequence[dict]) -> dict:
    items = ranker._items  # noqa: SLF001 - evaluation reads the catalog the ranker serves
    courses = [i for i in items.values() if i.format == "course"]
    by_popularity = [
        i.item_id for i in sorted(courses, key=lambda i: (-(i.popularity or 0), i.item_id))
    ]
    scores: dict[str, list[float]] = defaultdict(list)
    for history_rows, hidden in next_course_cases(rows):
        seen = {r["course_id"] for r in history_rows}
        history = [
            Event(user_id="u", item_id=r["course_id"],
                  type="complete" if r["status"] == 2 else "enroll", ts=r["enrolled_at"])
            for r in history_rows
        ]  # fmt: skip
        answer = ranker.recommend(Profile(), history, limit=50)
        shown = [e.item.item_id for e in answer.route + answer.more + answer.quick]
        ours = [i for i in dict.fromkeys(shown) if i in items and items[i].format == "course"]
        ours = [i for i in ours if i not in seen]
        # The ranker shows a mix of formats; missing course slots fall back to popularity.
        ours += [i for i in by_popularity if i not in seen and i not in ours]
        popular = [i for i in by_popularity if i not in seen]
        for name, ranked in (("ours", ours), ("popularity", popular)):
            scores[f"{name}_hit5"].append(hit_rate(ranked, {hidden}, 5))
            scores[f"{name}_mrr"].append(reciprocal_rank(ranked, {hidden}))
    count = len(scores["ours_hit5"])
    return (
        {"users": count, **{k: statistics.fmean(v) for k, v in scores.items()}}
        if count
        else {"users": 0}
    )


def evaluate_editorial(ranker: Ranker, pairs: Sequence[dict]) -> dict:
    items = ranker._items  # noqa: SLF001
    links: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        links[pair["source_id"]].add(pair["item_id"])
    by_topic: dict[str, list[str]] = defaultdict(list)
    for item in sorted(
        items.values(),
        key=lambda i: (-(i.published_at.timestamp() if i.published_at else 0), i.item_id),
    ):
        if item.topics:
            by_topic[item.topics[0]].append(item.item_id)
    scores: dict[str, list[float]] = defaultdict(list)
    for source in sorted(links):
        item = items[source]
        hits = ranker._index.search(f"{item.title}. {item.summary}", limit=K, excluded={source})  # noqa: SLF001
        ours = [hit.item_id for hit in hits]
        topic = item.topics[0] if item.topics else None
        same_topic = [i for i in by_topic.get(topic, []) if i != source][:K] if topic else []
        for name, ranked in (("index", ours), ("same_topic", same_topic)):
            scores[f"{name}_recall"].append(recall(ranked, links[source], K))
            scores[f"{name}_ndcg"].append(ndcg(ranked, links[source], K))
    return {"sources": len(links), **{k: statistics.fmean(v) for k, v in scores.items()}}


def evaluate_profiles(ranker: Ranker) -> dict:
    suite = load_profiles()
    result: dict[str, dict] = {}
    for mode in ("ours", "baseline"):
        formats, diversity, latency, shown_ids = [], [], [], set()
        clarified, clarify_cases, false_clarify = 0, 0, 0
        for case in suite.profiles:
            started = time.perf_counter()
            answer = ranker.recommend(case.profile, [], mode=mode, limit=K)
            latency.append(time.perf_counter() - started)
            cards = answer.route + answer.more
            top = [e.item for e in cards[:5]]
            shown_ids |= {e.item.item_id for e in cards[:K]}
            if top:
                formats.append(len({i.format for i in top}))
                pairs = [(a, b) for n, a in enumerate(top) for b in top[n + 1 :]]
                same = [(a.topics[:1] == b.topics[:1]) for a, b in pairs]
                diversity.append(1 - statistics.fmean(same) if same else 1.0)
            if case.expected_behavior == "clarify":
                clarify_cases += 1
                clarified += answer.clarification is not None
            elif answer.clarification is not None:
                false_clarify += 1
        result[mode] = {
            "formats_top5": statistics.fmean(formats) if formats else 0.0,
            "diversity_top5": statistics.fmean(diversity) if diversity else 0.0,
            "coverage": len(shown_ids) / len(ranker._items),  # noqa: SLF001
            "clarified": f"{clarified} из {clarify_cases}",
            "false_clarify": false_clarify,
            "latency_p95_ms": sorted(latency)[int(0.95 * (len(latency) - 1))] * 1000,
        }
    return result


def render(next_course: dict, editorial: dict, profiles: dict) -> str:
    f = lambda value: f"{value:.3f}"  # noqa: E731
    lines = [
        "# Офлайн-оценка: этап 1",
        "",
        "Отчёт создаёт `python -m alfarec.eval` по каталогу, индексу B04, ранжированию B05 и"
        " входам из `data/handoff/` (см. [отчёт передачи](b02_handoff.md)). Это проверки без"
        " ручной разметки. Главная метрика ТЗ — nDCG@10 на тестовых профилях по оценкам 0/1/2 —"
        " ещё не посчитана: нужны оценки двух участников. Веса ранжирования ручные, абляций нет.",
        "",
        "## Следующий курс по поведению",
        "",
    ]
    if next_course.get("users"):
        lines += [
            f"Пользователей с однозначным последним курсом: {next_course['users']}. Последний по"
            " дате курс скрыт, остальные переданы как история; даты точны до месяца, поэтому"
            " пользователи с несколькими курсами в последнем месяце исключены.",
            "",
            "| Метод | HitRate@5 | MRR |",
            "|---|---|---|",
            f"| Наша система (курсы из выдачи, остаток — по популярности) | {f(next_course['ours_hit5'])} | {f(next_course['ours_mrr'])} |",  # noqa: E501
            f"| Популярность | {f(next_course['popularity_hit5'])} | {f(next_course['popularity_mrr'])} |",  # noqa: E501
            "",
            "Популярность посчитана по всем записям каталога, включая скрытые; это даёт ей"
            " небольшое преимущество, которое не исправлялось.",
            "",
            "Наша система здесь проигрывает популярности. При пустом профиле курсы в выдаче"
            " подбираются по текстовой близости к пройденным, а популярность весит мало; для"
            " следующего курса этого недостаточно. Это задача подбора весов и графа совместных"
            " записей (B04, B06), а не исправление под эту метрику.",
        ]
    lines += [
        "",
        "## Редакторские подборки",
        "",
        f"Исходных материалов: {editorial['sources']}. По названию и описанию источника ищутся"
        " связанные материалы из `RecommendedItems` и `DocumentBindings`.",
        "",
        "| Метод | Recall@10 | nDCG@10 |",
        "|---|---|---|",
        f"| Индекс TF-IDF B04 по метаданным | {f(editorial['index_recall'])} | {f(editorial['index_ndcg'])} |",  # noqa: E501
        f"| Свежие материалы той же темы | {f(editorial['same_topic_recall'])} | {f(editorial['same_topic_ndcg'])} |",  # noqa: E501
        "",
        "## Списки по 36 тестовым профилям (без разметки)",
        "",
        "| Метрика | Наша система | Бейзлайн |",
        "|---|---|---|",
    ]
    ours, base = profiles["ours"], profiles["baseline"]
    rows = [
        ("Форматов в топ-5, среднее", f(ours["formats_top5"]), f(base["formats_top5"])),
        (
            "Разнообразие тем в топ-5 (доля пар с разной темой)",
            f(ours["diversity_top5"]),
            f(base["diversity_top5"]),
        ),
        ("Покрытие каталога топ-10 по всем профилям", f(ours["coverage"]), f(base["coverage"])),
        ("Уточнение на профилях «уточнить»", ours["clarified"], base["clarified"]),
        (
            "Лишние уточнения на остальных профилях",
            str(ours["false_clarify"]),
            str(base["false_clarify"]),
        ),
        (
            "Задержка ответа p95, мс (локально, CPU)",
            f"{ours['latency_p95_ms']:.0f}",
            f"{base['latency_p95_ms']:.0f}",
        ),
    ]
    lines += [f"| {name} | {a} | {b} |" for name, a, b in rows]
    lines += [
        "",
        "Цифры описывают форму выдачи, а не её релевантность: качество подтверждается только"
        " ручной разметкой. Задержка зависит от машины и в git меняется при каждом прогоне.",
        "",
        "## Не сделано на этом этапе",
        "",
        "- Пул выдач для ручной разметки, оценки 0/1/2, каппа Коэна, nDCG@10, Precision@5 и MRR"
        " на профилях `holdout`.",
        "- Абляции (стадия и уровень, история, переранжирование) и подбор весов на `tune`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    needed = [args.handoff / n for n in ("course_enrolments.parquet", "editorial_pairs.parquet")]
    if not all(path.is_file() for path in needed):
        print(f"Handoff files not found in {args.handoff}; run export_handoff", file=sys.stderr)
        return 1
    ranker = load_ranker()
    next_course = evaluate_next_course(ranker, pq.read_table(needed[0]).to_pylist())
    editorial = evaluate_editorial(ranker, pq.read_table(needed[1]).to_pylist())
    profiles = evaluate_profiles(ranker)
    args.output.write_text(render(next_course, editorial, profiles), encoding="utf-8", newline="\n")
    print(f"Report: {args.output}")
    return 0
