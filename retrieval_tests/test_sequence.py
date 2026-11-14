from datetime import UTC, datetime, timedelta

import pytest

from alfarec.eval.offline import split_users
from alfarec.recsys.contracts import Event, Profile, ReasonCode
from alfarec.recsys.ranking import load_ranker
from alfarec.recsys.sequence import CourseTransitions

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def row(user: str, course: str, month: int) -> dict:
    return {
        "user_key": user,
        "course_id": course,
        "enrolled_at": datetime(2025, month, 1, tzinfo=UTC),
    }


def test_pairs_follow_enrolment_order_and_rare_pairs_are_dropped():
    rows = [
        row("a", "c1", 1), row("a", "c2", 2), row("a", "c2", 3),
        row("b", "c1", 1), row("b", "c2", 4),
        row("c", "c2", 1), row("c", "c3", 2),  # seen once: below the privacy threshold
        row("d", "c1", 5), row("d", "c3", 5),  # same month: no order
    ]  # fmt: skip
    counts = CourseTransitions.build(rows).counts
    # A user counts once per pair, however many times they re-enrolled.
    assert counts == {"c1": {"c2": 2}}


def test_scores_prefer_the_followed_course_and_name_its_source():
    transitions = CourseTransitions({"c1": {"c3": 10}})
    # c2 is twice as popular, but c3 is what people take after c1.
    scores = transitions.scores(["c1"], {"c2": 20.0, "c3": 10.0})
    assert scores["c3"][0] == 1.0 and scores["c3"][1] == "c1"
    assert scores["c2"][0] < 1.0 and scores["c2"][1] == ""


def test_scores_without_history_are_empty():
    assert CourseTransitions({"c1": {"c2": 3}}).scores([], {"c2": 1.0}) == {}


def test_round_trip(tmp_path):
    transitions = CourseTransitions({"c1": {"c2": 3}})
    path = tmp_path / "t.json"
    transitions.save(path)
    assert CourseTransitions.load(path) == transitions


def test_split_users_keeps_each_user_on_one_side():
    rows = [row(f"u{n}", "c1", 1) for n in range(200)] * 2
    tune, holdout = split_users(rows)
    assert {r["user_key"] for r in tune}.isdisjoint({r["user_key"] for r in holdout})
    assert 0.45 < len(tune) / len(rows) < 0.75


@pytest.fixture(scope="module")
def courses():
    ranker = load_ranker(transitions_path=None)
    items = ranker._items  # noqa: SLF001
    return sorted(
        (i for i in items.values() if i.format == "course" and (i.popularity or 0) > 0),
        key=lambda i: (i.popularity, i.item_id),
    )


def test_ranker_puts_the_followed_course_first_and_explains_it(courses):
    source, rare = courses[-1], courses[0]
    ranker = load_ranker(transitions=CourseTransitions({source.item_id: {rare.item_id: 50}}))
    history = [
        Event(user_id="u", item_id=source.item_id, type="enroll", ts=NOW - timedelta(days=3))
    ]
    answer = ranker.recommend(Profile(), history, limit=50)
    cards = [c for c in answer.more if not c.next_step.label == "Продолжить курс"]
    shown_courses = [c for c in cards if c.item.format == "course"]
    assert shown_courses[0].item.item_id == rare.item_id
    # "Taken after X" is not "continues X": the reason says similar people chose it.
    assert ReasonCode.POPULAR_WITH_SIMILAR in {r.code for r in shown_courses[0].reasons}


def test_stated_tasks_keep_unrelated_follow_up_courses_down(courses):
    source, rare = courses[-1], courses[0]
    ranker = load_ranker(transitions=CourseTransitions({source.item_id: {rare.item_id: 50}}))
    facts = ranker._facts  # noqa: SLF001
    task = next(
        t for c in reversed(courses[1:-1]) for t, s in facts[c.item_id].tasks.items()
        if s >= 0.8 and t not in facts[rare.item_id].tasks
    )  # fmt: skip
    history = [Event(user_id="u", item_id=source.item_id, type="enroll", ts=NOW)]
    answer = ranker.recommend(Profile(tasks=[task]), history, limit=10)
    top = [c.item.item_id for c in answer.route + answer.more[:3]]
    assert rare.item_id not in top


def test_ranker_without_transitions_ignores_the_feature(courses):
    source = courses[-1]
    ranker = load_ranker(transitions_path=None)
    history = [Event(user_id="u", item_id=source.item_id, type="enroll", ts=NOW)]
    answer = ranker.recommend(Profile(), history, limit=10)
    assert all("sequence" not in r.params for c in answer.more for r in c.reasons)
