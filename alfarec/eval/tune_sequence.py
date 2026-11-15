"""Tune the course-transition signal on tune users only (inner split), never on holdout.

Usage: python -m alfarec.eval.tune_sequence [--handoff DIR] [--output PATH]

Tune users (60%, see offline.split_users) are split again by the same hash: buckets 0-2 build
the transitions, buckets 3-5 are scored. Step 1 picks the popularity mix and smoothing on the
transition signal alone; step 2 picks the ranker weight of the `sequence` feature.
"""

import argparse
import statistics
import zlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from alfarec.eval.offline import (
    DEFAULT_HANDOFF,
    evaluate_next_course,
    hit_rate,
    next_course_cases,
    reciprocal_rank,
    split_users,
)
from alfarec.recsys.ranking import load_ranker
from alfarec.recsys.sequence import CourseTransitions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "sequence_tuning.md"
BLENDS = (0.3, 0.5, 0.8, 1.0)
SMOOTHINGS = (1.0, 5.0, 20.0)
WEIGHTS = (2.0, 4.0, 8.0, 16.0)
TOLERANCE = 0.01


def inner_split(tune: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    build, check = [], []
    for row in tune:
        (build if zlib.crc32(row["user_key"].encode("utf-8")) % 10 < 3 else check).append(row)
    return build, check


def signal_only(transitions: CourseTransitions, check: Sequence[dict]) -> tuple[float, float]:
    """HitRate@5 and MRR of ranking courses by the transition score alone."""
    hits, ranks = [], []
    courses = sorted(transitions.popularity)
    for case in next_course_cases(check):
        seen = {event.item_id for event in case.history}
        scores = transitions.scores(
            [event.item_id for event in case.history],
            {c: 0.0 for c in courses if c not in seen},
        )
        ranked = sorted(scores, key=lambda c: (-scores[c][0], c))
        hits.append(hit_rate(ranked, {case.hidden}, 5))
        ranks.append(reciprocal_rank(ranked, {case.hidden}))
    return statistics.fmean(hits), statistics.fmean(ranks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    rows = pq.read_table(args.handoff / "course_enrolments.parquet").to_pylist()
    tune, _ = split_users(rows)
    build, check = inner_split(tune)
    base = CourseTransitions.build(build)
    lines = [
        "# Подбор сигнала переходов между курсами",
        "",
        "Создано `python -m alfarec.eval.tune_sequence`. Только пользователи `tune`: 30% строят"
        " переходы и популярность, 30% оцениваются; `holdout` не читается.",
        "",
        "## Шаг 1. Смесь с популярностью и сглаживание (только сигнал переходов)",
        "",
        "| Доля переходов | Сглаживание | HitRate@5 | MRR |",
        "|---|---|---|---|",
    ]
    best = (-1.0, 0.0, 0.0)
    for blend in BLENDS:
        for smoothing in SMOOTHINGS:
            hit, mrr = signal_only(replace(base, blend=blend, smoothing=smoothing), check)
            lines.append(f"| {blend} | {smoothing} | {hit:.3f} | {mrr:.3f} |")
            best = max(best, (mrr, blend, smoothing))
    _, blend, smoothing = best
    lines += ["", f"Выбрано по MRR: доля {blend}, сглаживание {smoothing}.", ""]
    lines += [
        "## Шаг 2. Вес признака `sequence` в ранжировании",
        "",
        "| Вес | HitRate@5 | MRR |",
        "|---|---|---|",
    ]
    chosen = replace(base, blend=blend, smoothing=smoothing)
    results: dict[float, float] = {}
    for weight in WEIGHTS:
        ranker = load_ranker(
            weights={"sequence": weight}, transitions_path=None, transitions=chosen
        )
        metrics = evaluate_next_course(ranker, check)
        lines.append(f"| {weight} | {metrics['ours_hit5']:.3f} | {metrics['ours_mrr']:.3f} |")
        results[weight] = metrics["ours_mrr"]
        print(f"weight {weight}: {metrics['ours_hit5']:.3f} {metrics['ours_mrr']:.3f}", flush=True)
    # Within noise of the best, the smallest weight keeps content signals in play.
    top = max(results.values())
    weight = min(w for w, mrr in results.items() if mrr >= top - TOLERANCE)
    lines += ["", f"Выбран наименьший вес в пределах {TOLERANCE} от лучшего MRR: {weight}.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
