from datetime import UTC, datetime

import pytest

from alfarec.eval.offline import (
    evaluate_editorial,
    evaluate_profiles,
    hit_rate,
    ndcg,
    next_course_cases,
    recall,
    reciprocal_rank,
)
from alfarec.recsys.ranking import load_ranker


def test_metrics_on_known_lists():
    ranked = ["a", "b", "c", "d"]
    assert hit_rate(ranked, {"c"}, 2) == 0.0 and hit_rate(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert recall(ranked, {"a", "x"}, 4) == 0.5
    assert ndcg(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert ndcg(["x", "a"], {"a"}, 10) == pytest.approx(1 / 1.5849625007211562)
    assert ndcg([], set(), 10) == 0.0


def test_next_course_cases_skip_ties_and_single_courses():
    def row(user, course, month):
        return {"user_key": user, "course_id": course, "status": 1,
                "enrolled_at": datetime(2026, month, 1, tzinfo=UTC)}  # fmt: skip

    rows = [
        row("u1", "c1", 1), row("u1", "c2", 3),
        row("u2", "c1", 2), row("u2", "c2", 2),
        row("u3", "c1", 1),
    ]  # fmt: skip
    cases = next_course_cases(rows)
    assert [(len(history), hidden) for history, hidden in cases] == [(1, "c2")]


@pytest.fixture(scope="module")
def ranker():
    return load_ranker()


def test_editorial_and_profile_checks_run_on_the_catalog(ranker):
    items = sorted(ranker._items.values(), key=lambda i: i.item_id)  # noqa: SLF001
    pairs = [{"source_id": items[0].item_id, "item_id": items[1].item_id}]
    editorial = evaluate_editorial(ranker, pairs)
    assert editorial["sources"] == 1
    assert 0 <= editorial["index_recall"] <= 1
    profiles = evaluate_profiles(ranker)
    assert profiles["ours"]["clarified"] == "5 из 5"
    assert profiles["ours"]["false_clarify"] == 0
    assert profiles["ours"]["formats_top5"] >= profiles["baseline"]["formats_top5"]
