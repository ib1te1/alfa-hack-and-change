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
from alfarec.recsys.taxonomy import load_taxonomy

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SUITE = load_profiles()
TAXONOMY = load_taxonomy()


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
        durations, answers = [], []
        for _ in range(3):
            started = time.perf_counter()
            answers.append(ranker.recommend(case.profile, [], limit=10))
            durations.append(time.perf_counter() - started)
        # The best of three runs keeps a busy CI machine from failing the check.
        assert min(durations) < 1.0, case.id
        assert answers[0] == answers[1] == answers[2], case.id


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


def grid():
    for task in TAXONOMY.tasks:
        for budget in (None, "quick", "hour", "course"):
            for experience in (None, "beginner", "advanced"):
                yield Profile(tasks=[task], time_budget=budget, experience=experience)


def test_top_five_respects_paid_quota_and_formats(ranker):
    profiles = [case.profile for case in SUITE.profiles] + list(grid())
    for profile in profiles:
        top = ranker.recommend(profile, [], limit=10).more[:5]
        assert sum(entry.item.access != "free" for entry in top) <= MAX_PAID_IN_TOP, profile
        if len(top) == 5:
            assert len({entry.item.format for entry in top}) >= 2, profile


def test_every_card_has_a_reason(ranker):
    profiles = [Profile()] + [case.profile for case in SUITE.profiles] + list(grid())[::5]
    for profile in profiles:
        answer = ranker.recommend(profile, [], limit=10)
        for entry in answer.route + answer.more + answer.quick:
            assert entry.reasons, (profile, entry.item.title)


def test_task_reasons_name_only_linked_tasks(ranker):
    answer = ranker.recommend(Profile(tasks=["franchising"]), [], limit=20)
    for entry in answer.route + answer.more:
        for reason in entry.reasons:
            if reason.code == "task_match":
                assert reason.params["task"] == "franchising"


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
    assert personal.more[0].next_step.label == "Продолжить курс"
    assert ids(personal.more) != ids(plain.more)


def test_known_and_dislike_remove_items_and_raise_level(ranker):
    profile = Profile(tasks=["taxes_reporting"], experience="beginner")
    first = ranker.recommend(profile, [], limit=10)
    known, disliked = first.more[0].item.item_id, first.more[1].item.item_id
    after = ranker.recommend(profile, [event(known, "known"), event(disliked, "dislike")])
    shown = ids(after.route) + ids(after.more) + ids(after.quick)
    assert known not in shown and disliked not in shown
    assert after.profile_used.experience == "intermediate"


@pytest.mark.parametrize("kind", ["known", "dislike", "unknown-item"])
def test_cold_start_survives_history_without_positive_events(ranker, kind):
    first = ranker.recommend(Profile(), []).more[0].item.item_id
    history = [event("not-in-catalog", "view")] if kind == "unknown-item" else [event(first, kind)]
    answer = ranker.recommend(Profile(), history)
    assert len(answer.route) == 5 and answer.quick
    assert first not in ids(answer.more) or kind == "unknown-item"


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
        ("Открываю магазин игрушек, как продвигать", ["business_start", "marketing"], None),
        ("Как получить ипотеку на коммерческую недвижимость", [], None),
        ("Работаю ботаником", [], None),
        ("Нужно 115 тысяч рублей", [], None),
        ("Заказали банкет на юбилей", [], None),
        ("Поставки из Европы", ["international"], None),
        ("Экспорт в Казахстан", ["international"], None),
        ("Кафе грузинской кухни", [], None),
        ("Как работать с 115-ФЗ", ["banking_compliance"], None),
    ],
)
def test_free_text_understanding(text, tasks, boundary):
    result = understand(text, load_intents())
    if tasks is not None:
        assert result.tasks == tasks
    assert (result.boundary.reason if result.boundary else None) == boundary


def test_profile_tasks_win_over_scope_boundaries(ranker):
    profile = Profile(tasks=["marketing"], free_text="Открываю магазин игрушек, как продвигать")
    answer = ranker.recommend(profile, [])
    assert answer.clarification is None and answer.route


def test_invalid_arguments_are_rejected(ranker):
    with pytest.raises(ValueError):
        ranker.recommend(Profile(), [], limit=0)
    with pytest.raises(ValueError):
        ranker.recommend(Profile(), [], mode="other")
