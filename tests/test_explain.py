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
            "Продолжает «Маркетплейсы», который вы уже изучали",
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
