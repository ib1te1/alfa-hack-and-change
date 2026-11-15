import time
from datetime import UTC, datetime, timedelta

import pytest

from alfarec.eval.profiles import load_profiles
from alfarec.recsys.contracts import Event, Profile, Recommender
from alfarec.recsys.ranking import (
    MAX_PAID_IN_TOP,
    ROUTE_ROLES,
    Scored,
    load_intents,
    load_ranker,
    understand,
)
from alfarec.recsys.taxonomy import load_taxonomy

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SUITE = load_profiles()
TAXONOMY = load_taxonomy()


def test_mmr_computes_each_candidate_selected_pair_once(ranker, monkeypatch):
    from collections import Counter

    original = ranker._similarity
    calls = Counter()

    def measured(left, right):
        calls[(left.item_id, right.item_id)] += 1
        return original(left, right)

    monkeypatch.setattr(ranker, "_similarity", measured)
    scored = [Scored(item, 50.0 - n) for n, item in enumerate(list(ranker._items.values())[:40])]
    selected = ranker._rerank(scored[2:], 20, prefix=scored[:2])
    assert len(selected) == 20
    assert len({entry.item.item_id for entry in selected}) == 20
    assert calls and max(calls.values()) == 1


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


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_continuing_course_counts_towards_paid_quota(ranker, limit):
    started = "8be9c4dc-d301-4af3-9548-0a091b8cf441"
    answer = ranker.recommend(
        Profile(tasks=["marketing"], time_budget="course"),
        [event(started, "enroll")],
        limit=limit,
    )
    assert answer.more[0].item.item_id == started
    assert answer.more[0].next_step.label == "Продолжить курс"
    assert len(answer.more) <= limit
    assert sum(e.item.access != "free" for e in answer.more[:5]) <= MAX_PAID_IN_TOP


def test_enroll_on_an_article_does_not_create_a_continue_course_action(ranker):
    article = "4cbddec7-c4ce-4442-87fa-c402f26924fb"
    answer = ranker.recommend(
        Profile(tasks=["banking_compliance"], time_budget="course"),
        [event(article, "enroll")],
    )
    assert sum(e.item.access != "free" for e in answer.more[:5]) <= MAX_PAID_IN_TOP
    assert all(
        e.next_step.label != "Продолжить курс" for e in answer.route + answer.more + answer.quick
    )


def test_paid_quota_is_preserved_when_only_paid_candidates_remain(ranker):
    from alfarec.recsys.ranking import Scored

    paid = [item for item in ranker._items.values() if item.access != "free"][:3]
    prefix = [Scored(paid[0], 1.0)]
    selected = ranker._rerank([Scored(item, 0.5) for item in paid[1:]], 4, prefix)
    assert selected == []


@pytest.mark.parametrize("task", ["marketing", "marketplaces", "business_start"])
def test_continuing_route_course_is_shown_exactly_once(ranker, task):
    profile = Profile(tasks=[task], time_budget="course")
    plain = ranker.recommend(profile, [])
    started = next(e.item.item_id for e in plain.route if e.item.format == "course")
    answer = ranker.recommend(profile, [event(started, "enroll")])
    shown = ids(answer.route + answer.more + answer.quick)
    assert shown.count(started) == 1
    assert len(shown) == len(set(shown))
    assert answer.more[0].item.item_id == started
    assert answer.more[0].next_step.label == "Продолжить курс"


@pytest.mark.parametrize("kind,value", [("rate", 3.0), ("rate", 1.0), ("dislike", None)])
def test_nonpositive_history_keeps_out_of_scope_clarification(ranker, kind, value):
    history = Event(
        user_id="u",
        item_id="000ecbf2-ccb2-4305-82e2-f3ed7ce1b22c",
        type=kind,
        value=value,
        ts=NOW,
    )
    profile = Profile(free_text="Рассчитать траекторию космического корабля")
    assert ranker.recommend(profile, []).clarification is not None
    answer = ranker.recommend(profile, [history])
    assert answer.clarification is not None
    assert answer.profile_used.tasks == []
    assert answer.more


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


@pytest.mark.parametrize("profile", [Profile(), Profile(free_text="How to start?")])
def test_baseline_accepts_profiles_without_selected_tasks(ranker, profile):
    answer = ranker.recommend(profile, [], mode="baseline", limit=5)
    assert len(answer.more) == 5
    assert answer.mode == "baseline"
    assert all(not entry.reasons for entry in answer.more)


@pytest.mark.parametrize("kind", ["known", "dislike", "complete"])
def test_cold_start_feedback_excludes_quick_cards(ranker, kind):
    first = ranker.recommend(Profile(), [])
    item_id = first.quick[0].item.item_id
    answer = ranker.recommend(Profile(), [event(item_id, kind)])
    assert item_id not in ids(answer.route + answer.more + answer.quick)


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
