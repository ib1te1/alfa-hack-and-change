import time
from datetime import UTC, datetime, timedelta

import pytest

from alfarec.eval.profiles import load_profiles
from alfarec.recsys.contracts import Event, Profile, Recommender
from alfarec.recsys.ranking import (
    MAX_PAID_IN_TOP,
    ROUTE_ROLES,
    load_intents,
    load_ranker,
    understand,
)

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SUITE = load_profiles()


@pytest.fixture(scope="module")
def ranker():
    return load_ranker()


def ids(entries) -> list[str]:
    return [entry.item.item_id for entry in entries]


def event(item_id: str, kind: str, days_ago: int = 1, user: str = "u") -> Event:
    return Event(user_id=user, item_id=item_id, type=kind, ts=NOW - timedelta(days=days_ago))


def test_ranker_implements_the_contract(ranker):
    recommender: Recommender = ranker
    answer = recommender.recommend(Profile(tasks=["marketing"]), [], "ours", 3)
    assert answer.mode == "ours" and len(answer.more) <= 3


def test_every_profile_is_answered_fast_and_deterministically(ranker):
    for case in SUITE.profiles:
        started = time.perf_counter()
        first = ranker.recommend(case.profile, [], limit=10)
        assert time.perf_counter() - started < 1.0, case.id
        assert first == ranker.recommend(case.profile, [], limit=10), case.id


@pytest.mark.parametrize("case", [c for c in SUITE.profiles if c.expected_behavior == "clarify"],
                         ids=lambda c: c.id)  # fmt: skip
def test_unclear_and_out_of_scope_requests_get_a_clarification(ranker, case):
    answer = ranker.recommend(case.profile, [])
    assert answer.clarification is not None
    assert 2 <= len(answer.clarification.options) <= 4
    assert answer.route == []
    assert answer.more, "a general selection accompanies the question"


@pytest.mark.parametrize("case", [c for c in SUITE.profiles if c.expected_behavior == "recommend"],
                         ids=lambda c: c.id)  # fmt: skip
def test_answerable_profiles_get_recommendations_without_clarification(ranker, case):
    answer = ranker.recommend(case.profile, [])
    assert answer.clarification is None
    assert answer.profile_used.tasks, "free text is mapped to tasks"
    assert answer.more


def test_a02_task_gives_an_ordered_route_on_topic(ranker):
    answer = ranker.recommend(Profile(tasks=["marketing"], industries=["services"]), [])
    assert 3 <= len(answer.route) <= 5
    roles = [ROUTE_ROLES[entry.item.format] for entry in answer.route]
    order = {"understand": 0, "do": 1, "deepen": 2}
    assert roles == sorted(roles, key=order.__getitem__)
    assert len({entry.item.format for entry in answer.route}) >= 2
    for entry in answer.route:
        assert entry.reasons[0].code == "task_match"
        assert entry.next_step.url == entry.item.url
    assert not set(ids(answer.route)) & set(ids(answer.more))


def test_top_five_respects_paid_quota_and_formats(ranker):
    for case in SUITE.profiles:
        top = ranker.recommend(case.profile, [], limit=10).more[:5]
        assert sum(entry.item.access != "free" for entry in top) <= MAX_PAID_IN_TOP, case.id


def test_a03_history_excludes_completed_and_continues_courses(ranker):
    profile = Profile(tasks=["marketplaces"], industries=["ecom"], time_budget="course")
    plain = ranker.recommend(profile, [], limit=20)
    done = next(e for e in plain.more + plain.route if e.item.format == "course").item.item_id
    started = next(
        e.item.item_id
        for e in plain.more
        if e.item.format == "course" and e.item.access == "free" and e.item.item_id != done
    )
    history = [event(done, "complete", 30), event(started, "enroll", 3)]
    personal = ranker.recommend(profile, history, limit=20)
    shown = ids(personal.route) + ids(personal.more) + ids(personal.quick)
    assert done not in shown
    assert personal.more[0].item.item_id == started
    assert personal.more[0].reasons[0].code == "continues_history"
    assert ids(personal.more) != ids(plain.more)


def test_known_and_dislike_remove_items_and_raise_level(ranker):
    profile = Profile(tasks=["taxes_reporting"], experience="beginner")
    first = ranker.recommend(profile, [], limit=10)
    known, disliked = first.more[0].item.item_id, first.more[1].item.item_id
    after = ranker.recommend(profile, [event(known, "known"), event(disliked, "dislike")])
    shown = ids(after.route) + ids(after.more) + ids(after.quick)
    assert known not in shown and disliked not in shown
    assert after.profile_used.experience == "intermediate"


def test_a04_cold_start_has_starter_set_and_week_plan(ranker):
    answer = ranker.recommend(Profile(), [])
    assert answer.clarification is None
    assert len(answer.route) == 5
    assert all(entry.item.duration_min and entry.item.duration_min <= 10 for entry in answer.route)
    assert 6 <= len(answer.more) <= 8
    assert len({entry.item.format for entry in answer.more}) >= 3
    assert answer.quick


def test_history_fills_an_empty_profile(ranker):
    course = next(e for e in ranker.recommend(Profile(tasks=["marketplaces"]), []).more
                  if e.item.format == "course")  # fmt: skip
    answer = ranker.recommend(Profile(), [event(course.item.item_id, "complete")])
    assert answer.profile_used.tasks


def test_baseline_mode_uses_topics_and_popularity(ranker):
    answer = ranker.recommend(Profile(tasks=["marketing"]), [], mode="baseline", limit=5)
    assert answer.mode == "baseline" and len(answer.more) == 5
    assert answer.route == [] and answer.quick == []
    popularity = [entry.item.popularity or 0 for entry in answer.more]
    assert popularity == sorted(popularity, reverse=True)


@pytest.mark.parametrize(
    ("text", "tasks", "boundary"),
    [
        ("Клиентов мало, как сделать так, чтобы обо мне узнали?", ["marketing"], None),
        ("Всё делаю сам, пора кого-нибудь нанять.", ["team_hiring"], None),
        ("Регистрация автосервиса в Казахстане", None, "foreign_law"),
        ("Как лечить зубную боль?", None, "off_topic"),
        ("Скажите точную выручку на следующий год", None, "forecast"),
        ("Хочу развиваться", [], None),
    ],
)
def test_free_text_understanding(text, tasks, boundary):
    result = understand(text, load_intents())
    if tasks is not None:
        assert result.tasks == tasks
    assert (result.boundary.reason if result.boundary else None) == boundary


def test_invalid_arguments_are_rejected(ranker):
    with pytest.raises(ValueError):
        ranker.recommend(Profile(), [], limit=0)
    with pytest.raises(ValueError):
        ranker.recommend(Profile(), [], mode="other")
