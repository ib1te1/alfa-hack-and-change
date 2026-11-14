import json

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from alfarec.api.main import DEFAULT_PERSONAS_PATH, build_services, create_app
from alfarec.pipeline.build_catalog import DEFAULT_DUMP, DEFAULT_INDEX_TEXT
from alfarec.pipeline.dump_reader import read_tables
from alfarec.pipeline.text import html_to_text
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import Reason, Recommendation
from alfarec.recsys.explain import explain_recommendation, next_step_for, reason_text, route_note
from alfarec.recsys.taxonomy import load_taxonomy

TAXONOMY = load_taxonomy()
CATALOG = load_catalog(DEFAULT_CATALOG_PATH)


def by_format(item_format: str):
    return next(item for item in CATALOG if item.format == item_format and item.access == "free")


def wrap(item, reasons=()):
    return Recommendation(
        item=item, score=0.0, reasons=list(reasons), next_step=next_step_for(item)
    )


@pytest.mark.parametrize(
    ("code", "params", "text"),
    [
        ("task_match", {"task": "marketing"}, "Подходит под вашу задачу «продвижение и маркетинг»"),
        ("task_match", {"task": "ai_automation"}, "Подходит под вашу задачу «ИИ и автоматизация»"),
        ("task_match", {}, "Подходит под вашу задачу"),
        ("industry_match", {"industry": "services"}, "Для сферы «Услуги»"),
        ("stage_fit", {"stage": "idea"}, "Подходит для этапа «думаю открыть»"),
        (
            "continues_history",
            {"title": "Маркетплейсы"},
            "По теме «Маркетплейсы» из вашей истории",
        ),
        ("short_format", {"minutes": 8}, "Займёт 8 мин"),
        ("short_format", {}, None),
        ("beginner_friendly", {}, "Для тех, кто только начинает"),
    ],
)
def test_reason_texts_use_only_params_and_taxonomy(code, params, text):
    assert reason_text(Reason(code=code, weight=1.0, params=params), TAXONOMY) == text


def test_every_reason_code_has_wording():
    from alfarec.recsys.contracts import ReasonCode

    for code in ReasonCode:
        params = {
            "task": "marketing",
            "industry": "services",
            "stage": "idea",
            "title": "Курс",
            "minutes": 5,
        }
        assert reason_text(Reason(code=code, weight=1.0, params=params), TAXONOMY)


def test_card_keeps_three_strongest_distinct_reasons():
    reasons = [
        Reason(code="fresh", weight=0.1),
        Reason(code="task_match", weight=0.9, params={"task": "marketing"}),
        Reason(code="short_format", weight=0.5),
        Reason(code="industry_match", weight=0.7, params={"industry": "services"}),
        Reason(code="beginner_friendly", weight=0.3),
        Reason(code="task_match", weight=0.2, params={"task": "marketing"}),
    ]
    explained = explain_recommendation(wrap(by_format("guide"), reasons), TAXONOMY)
    assert [reason.code for reason in explained.reasons] == [
        "task_match", "industry_match", "beginner_friendly",
    ]  # fmt: skip
    assert all(reason.text for reason in explained.reasons)


def test_route_note_follows_step_roles():
    route = [
        wrap(by_format("quick_answer")),
        wrap(by_format("template")),
        wrap(by_format("course")),
    ]
    assert route_note(route) == (
        "Сначала разберёмся в основах, затем перейдём к шаблонам и расчётам,"
        " после — закрепим системным материалом."
    )
    assert route_note(route[:1]) is None
    assert route_note([]) is None


def test_next_step_by_access():
    paid = next(item for item in CATALOG if item.access == "paid")
    plus = next(item for item in CATALOG if item.access == "subscription")
    assert (next_step_for(paid).kind, next_step_for(paid).label) == ("purchase", "Купить курс")
    assert next_step_for(plus).kind == "subscribe"
    assert next_step_for(by_format("template")).label == "Скачать шаблон"


def test_popularity_does_not_claim_a_similar_user_cohort():
    text = reason_text(Reason(code="popular_with_similar", weight=0.5), TAXONOMY)
    assert text == "Популярный материал на платформе"


def test_freshness_does_not_claim_current_legal_validity():
    text = reason_text(Reason(code="fresh", weight=0.2), TAXONOMY)
    assert text == "Материал об изменениях в законах"


@pytest.mark.parametrize(
    "code,key",
    [
        ("continues_history", "title"),
        ("task_match", "task"),
        ("stage_fit", "stage"),
        ("industry_match", "industry"),
    ],
)
def test_long_reason_params_produce_valid_serializable_cards(code, key):
    recommendation = wrap(
        by_format("guide"),
        [
            Reason(code=code, weight=1.0, params={key: "Д" * 200}),
        ],
    )
    explained = explain_recommendation(recommendation, TAXONOMY)
    validated = Recommendation.model_validate_json(explained.model_dump_json())
    assert len(validated.reasons) == 1
    assert len(validated.reasons[0].text) <= 200
    assert "…" in validated.reasons[0].text


def test_negative_contributions_are_not_presented_as_benefits():
    explained = explain_recommendation(
        wrap(
            by_format("guide"),
            [
                Reason(code="task_match", weight=-1.0, params={"task": "marketing"}),
                Reason(code="short_format", weight=0.0, params={"minutes": 5}),
            ],
        ),
        TAXONOMY,
    )
    assert [reason.code for reason in explained.reasons] == ["short_format"]


@pytest.mark.parametrize(
    "item_format,kind,label",
    [
        ("quick_answer", "read", "Читать"),
        ("guide", "read", "Читать"),
        ("template", "download", "Скачать шаблон"),
        ("tool", "calculate", "Посчитать"),
        ("course", "start_course", "Начать курс"),
        ("recording", "watch", "Смотреть запись"),
        ("podcast", "listen", "Слушать"),
    ],
)
@pytest.mark.parametrize("access", ["free", "paid", "subscription"])
def test_all_next_steps_preserve_the_source_page(item_format, kind, label, access):
    item = by_format("guide").model_copy(update={"format": item_format, "access": access})
    step = next_step_for(item)
    assert step.url == item.url
    if access == "subscription":
        assert (step.kind, step.label) == ("subscribe", "Оформить Курс+")
    elif access == "paid":
        assert step.kind == "purchase"
        assert step.label == ("Купить курс" if item_format == "course" else "Купить")
    else:
        assert (step.kind, step.label) == (kind, label)


def test_explaining_preserves_the_continue_course_action():
    item = by_format("course")
    from alfarec.recsys.contracts import NextStep

    recommendation = wrap(
        item, [Reason(code="continues_history", weight=1.0, params={"title": item.title})]
    )
    recommendation.next_step = NextStep(
        kind="start_course",
        label="Продолжить курс",
        url=item.url,
    )
    explained = explain_recommendation(recommendation, TAXONOMY)
    assert explained.next_step == recommendation.next_step


def test_api_can_serve_history_reason_with_max_length_title(tmp_path):
    from alfarec.recsys.contracts import RecommendationResponse

    recommendation = wrap(
        by_format("course"),
        [
            Reason(code="continues_history", weight=1.0, params={"title": "Я" * 200}),
        ],
    )

    class LongTitleRecommender:
        def recommend(self, profile, history, mode, limit):
            return RecommendationResponse(more=[recommendation], mode=mode, profile_used=profile)

    services = build_services(
        DEFAULT_CATALOG_PATH, tmp_path / "events.jsonl", DEFAULT_PERSONAS_PATH
    )
    services.recommender = LongTitleRecommender()
    with TestClient(create_app(services)) as api:
        answer = api.post("/api/recommendations", json={"profile": {}})
    assert answer.status_code == 200
    validated = RecommendationResponse.model_validate(answer.json())
    assert len(validated.more[0].reasons[0].text) <= 200


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    events = tmp_path_factory.mktemp("events") / "events.jsonl"
    services = build_services(DEFAULT_CATALOG_PATH, events, DEFAULT_PERSONAS_PATH)
    with TestClient(create_app(services)) as test_client:
        yield test_client


def paid_answers(client) -> list[str]:
    """API output that shows paid and subscription materials: cards and recommendations."""
    closed = [item for item in CATALOG if item.access != "free"]
    texts = [client.get(f"/api/items/{item.item_id}").text for item in closed]
    for task in TAXONOMY.tasks:
        body = {"profile": {"tasks": [task], "time_budget": "course"}, "limit": 50}
        for mode in ("ours", "baseline"):
            answer = client.post("/api/recommendations", json={**body, "mode": mode}).json()
            for block in ("route", "more", "quick"):
                texts += [json.dumps(entry, ensure_ascii=False) for entry in answer[block]
                          if entry["item"]["access"] != "free"]  # fmt: skip
    return texts


def test_paid_cards_show_only_public_fields(client):
    closed = [item for item in CATALOG if item.access != "free"]
    assert closed
    for item in closed:
        card = client.get(f"/api/items/{item.item_id}").json()
        assert set(card) == set(item.model_dump())
        assert len(card["summary"]) <= 300
        assert card["url"].startswith("https://")


@pytest.mark.skipif(not DEFAULT_DUMP.is_file(), reason="the dump is handed over outside git")
def test_api_never_returns_lesson_text_for_paid_materials(client):
    tables = read_tables(DEFAULT_DUMP, ["LessonEmbedItems"])
    lesson_lines = {
        line
        for entry in tables["LessonEmbedItems"]
        for line in html_to_text(entry["HtmlContent"]).splitlines()
        if len(line) >= 40
    }
    corpus = "\n".join(paid_answers(client))
    assert corpus
    assert not [line for line in lesson_lines if line in corpus]


@pytest.mark.skipif(not DEFAULT_INDEX_TEXT.is_file(), reason="index text is built from the dump")
def test_index_text_of_closed_materials_is_landing_only():
    rows = pq.read_table(DEFAULT_INDEX_TEXT).to_pylist()
    closed_articles = [
        row for row in rows if row["item_type"] == "article" and row["access"] != "free"
    ]
    assert closed_articles
    assert all(len(row["index_text"]) < 1500 for row in closed_articles)
