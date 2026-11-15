from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alfarec.eval.offline import (
    evaluate_editorial,
    evaluate_next_course,
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
    assert [(len(case.history), case.hidden) for case in cases] == [(1, "c2")]
    assert cases[0].cutoff == datetime(2026, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("completed_at", "status", "completed"),
    [
        (datetime(2026, 2, 1, tzinfo=UTC), 2, True),
        (datetime(2026, 3, 1, tzinfo=UTC), 2, False),
        (datetime(2026, 3, 15, tzinfo=UTC), 2, False),
        (datetime(2026, 10, 1, tzinfo=UTC), 2, False),
        (None, 2, False),
        (None, 1, False),
    ],
)
def test_completion_history_uses_real_dates_before_prediction(completed_at, status, completed):
    enrolled_at = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {"user_key": "u", "course_id": "old", "enrolled_at": enrolled_at,
         "completed_at": completed_at, "status": status},
        {"user_key": "u", "course_id": "hidden", "status": 1,
         "enrolled_at": datetime(2026, 3, 1, tzinfo=UTC)},
    ]  # fmt: skip
    case = next_course_cases(rows)[0]
    assert [(event.type, event.ts) for event in case.history] == (
        [("enroll", enrolled_at), ("complete", completed_at)]
        if completed
        else [("enroll", enrolled_at)]
    )
    assert all(event.item_id != case.hidden and event.ts < case.cutoff for event in case.history)


def test_next_course_skips_unknown_dates_same_month_and_repeated_target():
    def row(course, date):
        return {"user_key": "u", "course_id": course, "enrolled_at": date, "status": 1}

    jan = datetime(2026, 1, 1, tzinfo=UTC)
    mar = datetime(2026, 3, 1, tzinfo=UTC)
    assert not next_course_cases([row("a", jan), row("b", None)])
    assert not next_course_cases([row("a", mar), row("b", mar.replace(day=20))])
    assert not next_course_cases([row("a", jan), row("b", mar), row("a", mar.replace(month=4))])


def test_next_course_evaluation_does_not_pass_future_completion():
    seen_history = []

    def recommend(profile, history, limit):
        seen_history.extend(history)
        return SimpleNamespace(route=[], more=[], quick=[])

    ranker = SimpleNamespace(
        _items={item_id: SimpleNamespace(item_id=item_id, format="course", popularity=1)
                for item_id in ("old", "hidden")},
        recommend=recommend,
    )  # fmt: skip
    rows = [
        {"user_key": "u", "course_id": "old", "status": 2,
         "enrolled_at": datetime(2026, 1, 1, tzinfo=UTC),
         "completed_at": datetime(2026, 10, 1, tzinfo=UTC)},
        {"user_key": "u", "course_id": "hidden", "status": 1,
         "enrolled_at": datetime(2026, 3, 1, tzinfo=UTC)},
    ]  # fmt: skip
    assert evaluate_next_course(ranker, rows)["users"] == 1
    assert [(event.item_id, event.type) for event in seen_history] == [("old", "enroll")]


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
