"""Offline evaluation, stage 1 of B06 (spec section 9): checks that need no human labels.

Usage: python -m alfarec.eval [--handoff DIR] [--output PATH]

1. Next course: users with two or more courses; the latest course is hidden and predicted.
2. Editorial links: from a source material, retrieve its "see also" items.
3. Profile lists: answers for the 36 test profiles, ours against the baseline.
Relevance judgments, ablations and weight tuning are later stages and are not reported here.
"""

import argparse
import json
import math
import statistics
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from alfarec.eval.profiles import load_profiles
from alfarec.recsys.contracts import Event, Profile
from alfarec.recsys.ranking import Ranker, load_ranker
from alfarec.recsys.sequence import CourseTransitions

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


@dataclass(frozen=True)
class NextCourseCase:
    history: list[Event]
    hidden: str
    cutoff: datetime


def _month(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_course_cases(rows: Sequence[dict]) -> list[NextCourseCase]:
    """Predict at the start of the hidden enrolment month, using only earlier events."""
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["user_key"]].append(row)
    cases = []
    for user in sorted(by_user):
        if any(row["enrolled_at"] is None for row in by_user[user]):
            # A missing date could be the user's latest enrolment; do not invent a split.
            continue
        entries = sorted(by_user[user], key=lambda r: (r["enrolled_at"], r["course_id"]))
        if len({r["course_id"] for r in entries}) < 2:
            continue
        last = entries[-1]
        # Dates are precise to a month: a tie for the latest course has no true answer.
        cutoff = _month(last["enrolled_at"])
        if sum(_month(r["enrolled_at"]) == cutoff for r in entries) > 1:
            continue
        previous = entries[:-1]
        if any(row["course_id"] == last["course_id"] for row in previous):
            # The target must be a new course, rather than a repeated enrolment.
            continue
        history = []
        for row in previous:
            history.append(
                Event(user_id="u", item_id=row["course_id"], type="enroll", ts=row["enrolled_at"])
            )
            completed_at = row.get("completed_at")
            if (
                row["status"] == 2
                and completed_at is not None
                and row["enrolled_at"] <= completed_at
                and _month(completed_at) < cutoff
            ):
                history.append(
                    Event(user_id="u", item_id=row["course_id"], type="complete", ts=completed_at)
                )
        history.sort(key=lambda event: (event.ts, event.item_id, event.type))
        cases.append(NextCourseCase(history, last["course_id"], cutoff))
    return cases


def split_users(rows: Sequence[dict], tune_share: int = 6) -> tuple[list[dict], list[dict]]:
    """Stable user split: tune_share in 10 users fit the transitions, the rest are held out."""
    tune, holdout = [], []
    for row in rows:
        bucket = zlib.crc32(row["user_key"].encode("utf-8")) % 10
        (tune if bucket < tune_share else holdout).append(row)
    return tune, holdout


def evaluate_next_course(ranker: Ranker, rows: Sequence[dict]) -> dict:
    items = ranker._items  # noqa: SLF001 - evaluation reads the catalog the ranker serves
    courses = [i for i in items.values() if i.format == "course"]
    by_popularity = [
        i.item_id for i in sorted(courses, key=lambda i: (-(i.popularity or 0), i.item_id))
    ]
    scores: dict[str, list[float]] = defaultdict(list)
    for case in next_course_cases(rows):
        seen = {event.item_id for event in case.history}
        hidden = case.hidden
        answer = ranker.recommend(Profile(), case.history, limit=50)
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


def evaluate_judged() -> dict | None:
    """Metrics on the human judgments over the judged lists, or None before they exist."""
    from alfarec.eval import judgments

    if not (judgments.JUDGMENTS_PATH.exists() and judgments.LISTS_PATH.exists()):
        return None
    lists = json.loads(judgments.LISTS_PATH.read_text(encoding="utf-8"))
    pool = json.loads(judgments.POOL_PATH.read_text(encoding="utf-8"))
    if lists["pool_sha256"] != pool["pool_sha256"]:
        raise ValueError("Judged lists do not belong to the current pool")
    splits = {p["id"]: p["split"] for p in pool["profiles"]}
    return judgments.evaluate_judgments(lists["lists"], judgments.final_grades(), splits)


def render(next_course: dict, editorial: dict, profiles: dict, judged: dict | None = None) -> str:
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
            "Пользователи стабильно делятся по хешу псевдонима: 60% (`tune`) строят переходы"
            " «курс → следующий курс» и подбирают их вес, на остальных 40% (`holdout`) считается"
            f" таблица. Случаев в `holdout` с однозначным последним курсом: {next_course['users']}."
            " Последний по дате курс скрыт. Прогноз строится на начало месяца его записи."
            " История содержит enroll по дате записи и complete по фактической дате завершения"
            " только из более ранних месяцев. При неизвестной дате завершения complete не"
            " создаётся. Неоднозначный последний месяц, неизвестные даты записей и повторный"
            " целевой курс исключены.",
            "",
            "| Метод | HitRate@5 | MRR |",
            "|---|---|---|",
            f"| Наша система с переходами между курсами | {f(next_course['ours_hit5'])} | {f(next_course['ours_mrr'])} |",  # noqa: E501
            f"| Наша система без переходов (только содержание) | {f(next_course['content_hit5'])} | {f(next_course['content_mrr'])} |",  # noqa: E501
            f"| Популярность | {f(next_course['popularity_hit5'])} | {f(next_course['popularity_mrr'])} |",  # noqa: E501
            "",
            "Это snapshot-диагностика с историей as-of, а не полный временной backtest."
            " Популярность, качество и доступность материалов взяты из полного снимка каталога,"
            " включая период после прогноза. Величина влияния этой утечки агрегатов неизвестна;"
            " для временной оценки нужны исторические снимки всех признаков.",
            "",
            "Курсы из выдачи идут первыми, недостающие места дополняются по популярности."
            " Переходы и их априорная популярность считаются только по записям `tune`, поэтому"
            " скрытые курсы `holdout` в этот признак не попадают. Вес, смесь и сглаживание"
            " выбирает `python -m alfarec.eval.tune_sequence` внутри `tune`"
            " ([протокол](sequence_tuning.md)); на `holdout` ничего не подбиралось. Признак"
            " популярности ранжирования и строка «Популярность» по-прежнему используют полный"
            " снимок каталога.",
        ]
    else:
        lines += ["Нет подходящих случаев следующего курса; метрики не рассчитаны."]
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
        *render_judgments(judged),
    ]
    return "\n".join(lines) + "\n"


def render_judgments(judged: dict | None) -> list[str]:
    from alfarec.eval.judgments import POOLED, VARIANT_LABELS, VARIANTS

    lines = ["## Релевантность по ручной разметке", ""]
    skipped = [VARIANT_LABELS[v] for v in VARIANTS if v not in POOLED]
    unmeasured = (
        "В пул для разметки входят только наша система и бейзлайн (решение команды по объёму"
        " ручной работы). Абляции не измерены: "
        + ", ".join(label.lower() for label in skipped)
        + ". Их списки сохраняются, но метрики по частично оценённому пулу были бы смещены."
    )
    if not judged:
        return lines + [
            "Пул выдач для разметки готов (`python -m alfarec.eval.judgments pool`, форма —"
            " `artifacts/judgments/form.html`), оценок двух участников ещё нет. nDCG@10,"
            " Precision@5 и MRR нашей системы и бейзлайна появятся здесь после"
            " `python -m alfarec.eval.judgments merge`. Веса, кроме переходов между курсами,"
            " не подбирались.",
            "",
            unmeasured,
        ]
    lines += [
        "nDCG@10 с выигрышем 2^оценка − 1; Precision@5 и MRR считают релевантной оценку 2."
        " Неоценённые карточки пропускаются и показаны отдельно, нулём не считаются.",
        "",
        unmeasured,
        "",
    ]
    for split, title in (("holdout", "`holdout` — основное сравнение"), ("tune", "`tune`")):
        rows = judged.get(split, {})
        if not rows:
            continue
        profiles = next(iter(rows.values()))["profiles"]
        lines += [
            f"### {title}, профилей: {profiles:.0f}",
            "",
            "| Вариант | nDCG@10 | Precision@5 | MRR | Неоценённых в списке |",
            "|---|---|---|---|---|",
        ]
        for variant in POOLED:
            if variant in rows:
                m = rows[variant]
                lines.append(
                    f"| {VARIANT_LABELS[variant]} | {m['ndcg10']:.3f} | {m['p5']:.3f} |"
                    f" {m['mrr']:.3f} | {m['unjudged']:.1f} |"
                )
        lines.append("")
    return lines


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
    tune, holdout = split_users(pq.read_table(needed[0]).to_pylist())
    # Transitions come from tune users only, so a hidden holdout course never informs them.
    with_sequence = load_ranker(transitions_path=None, transitions=CourseTransitions.build(tune))
    content_only = load_ranker(transitions_path=None)
    next_course = evaluate_next_course(with_sequence, holdout)
    next_course["content_hit5"], next_course["content_mrr"] = (
        (m := evaluate_next_course(content_only, holdout)).get("ours_hit5", 0.0),
        m.get("ours_mrr", 0.0),
    )
    editorial = evaluate_editorial(ranker, pq.read_table(needed[1]).to_pylist())
    profiles = evaluate_profiles(ranker)
    report = render(next_course, editorial, profiles, evaluate_judged())
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Report: {args.output}")
    return 0
