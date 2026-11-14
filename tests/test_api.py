import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alfarec.api.main import build_services, create_app
from alfarec.recsys.catalog import save_catalog
from alfarec.recsys.contracts import Item


def make_item(item_id: str, **values: object) -> Item:
    payload = {
        "item_id": item_id,
        "item_type": "article",
        "format": "guide",
        "title": f"Материал {item_id}",
        "summary": "Публичное описание",
        "duration_min": 8,
        "access": "free",
        "url": f"https://example.test/articles/{item_id}",
        "topics": ["Маркетинг"],
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    payload.update(values)
    return Item.model_validate(payload)


CATALOG = [
    make_item("promo-guide", popularity=50.0),
    make_item("promo-quick", format="quick_answer", duration_min=3, popularity=10.0),
    make_item("tax-guide", topics=["Налоги и законы"], popularity=90.0),
    make_item(
        "plus-course", item_type="course", format="course", access="subscription",
        price_rub=990, topics=["Маркетинг"], url="https://example.test/courses/plus",
    ),
    make_item("ndfl", item_type="calculator", format="tool", duration_min=None, topics=[]),
]  # fmt: skip


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    catalog = tmp_path / "catalog.parquet"
    save_catalog(CATALOG, catalog)
    return {
        "catalog": tmp_path / "catalog.parquet",
        "events": tmp_path / "events" / "events.jsonl",
        "personas": tmp_path / "personas.json",
    }


@pytest.fixture
def client(paths: dict[str, Path]):
    services = build_services(paths["catalog"], paths["events"], paths["personas"])
    with TestClient(create_app(services)) as test_client:
        yield test_client


def recommend(client: TestClient, **body: object) -> dict:
    response = client.post("/api/recommendations", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def ids(entries: list[dict]) -> list[str]:
    return [entry["item"]["item_id"] for entry in entries]


def test_health_reports_loaded_catalog(client: TestClient) -> None:
    assert client.get("/api/health").json() == {
        "status": "ok",
        "ready": True,
        "artifacts_loaded": True,
        "catalog_items": 5,
        "encoder": "unavailable",
        "recommender": "interim",
    }


def test_missing_catalog_degrades_with_a_clear_message(tmp_path: Path) -> None:
    services = build_services(tmp_path / "none.parquet", None, tmp_path / "none.json")
    with TestClient(create_app(services)) as client:
        health = client.get("/api/health").json()
        assert (health["status"], health["ready"], health["catalog_items"]) == (
            "degraded",
            False,
            0,
        )
        response = client.post("/api/recommendations", json={})
        assert response.status_code == 503
        assert "Каталог не загружен" in response.json()["detail"]


def test_baseline_follows_task_topics_and_popularity(client: TestClient) -> None:
    result = recommend(client, profile={"tasks": ["marketing"]}, mode="baseline")
    assert result["mode"] == "baseline"
    assert ids(result["more"]) == ["promo-guide", "promo-quick", "plus-course"]
    assert result["more"][0]["reasons"][0]["code"] == "task_match"
    empty = recommend(client, mode="baseline", limit=2)
    assert ids(empty["more"]) == ["tax-guide", "promo-guide"]


def test_next_step_depends_on_format_and_access(client: TestClient) -> None:
    result = recommend(client, limit=10)
    steps = {entry["item"]["item_id"]: entry["next_step"] for entry in result["more"]}
    assert steps["plus-course"]["kind"] == "subscribe"
    assert steps["plus-course"]["label"] == "Оформить Курс+"
    assert steps["ndfl"]["kind"] == "calculate"
    assert steps["promo-guide"]["kind"] == "read"


def test_feedback_events_change_the_next_answer_and_persist(
    client: TestClient, paths: dict[str, Path]
) -> None:
    body = {"profile": {"tasks": ["marketing"]}, "user_id": "u1"}
    assert ids(recommend(client, **body)["more"])[0] == "promo-guide"
    event = {
        "user_id": "u1",
        "item_id": "promo-guide",
        "type": "known",
        "ts": "2026-11-14T10:00:00Z",
    }
    accepted = client.post("/api/events", json=event)
    assert accepted.status_code == 201
    assert accepted.json() == {"status": "accepted", "events_for_user": 1}
    assert "promo-guide" not in ids(recommend(client, **body)["more"])
    assert "promo-guide" in ids(recommend(client, **{**body, "user_id": "u2"})["more"])

    services = build_services(paths["catalog"], paths["events"], paths["personas"])
    assert [entry.item_id for entry in services.events.history("u1")] == ["promo-guide"]


def test_quick_block_and_time_budget(client: TestClient) -> None:
    result = recommend(client, profile={"tasks": ["marketing"], "time_budget": "quick"})
    assert ids(result["quick"]) == ["promo-quick"]
    assert result["quick"][0]["reasons"][-1] == {
        "code": "short_format",
        "weight": 0.0,
        "params": {"minutes": 3},
    }
    assert ids(result["more"]) == ["promo-quick"]
    course_first = recommend(client, profile={"tasks": ["marketing"], "time_budget": "course"})
    assert ids(course_first["more"])[0] == "plus-course"


def test_invalid_requests_get_readable_errors(client: TestClient) -> None:
    unknown = client.post("/api/recommendations", json={"profile": {"tasks": ["astrology"]}})
    assert unknown.status_code == 422
    assert "astrology" in unknown.json()["detail"]
    missing = client.post(
        "/api/events",
        json={"user_id": "u1", "item_id": "nope", "type": "view", "ts": "2026-11-14T10:00:00Z"},
    )
    assert missing.status_code == 404
    assert client.get("/api/items/nope").status_code == 404


def test_item_card_and_meta(client: TestClient) -> None:
    card = client.get("/api/items/plus-course").json()
    assert card["access"] == "subscription" and card["price_rub"] == 990
    meta = client.get("/api/meta").json()
    assert {entry["id"] for entry in meta["stages"]} == {"idea", "launch", "operating", "growth"}
    assert "marketing" in {entry["id"] for entry in meta["tasks"]}
    assert [entry["id"] for entry in meta["time_budgets"]] == ["quick", "hour", "course"]
    assert client.get("/api/personas").json() == []


def test_personas_seed_their_history(paths: dict[str, Path]) -> None:
    persona = {
        "id": "seller",
        "title": "ИП на маркетплейсах",
        "user_id": "persona-seller",
        "profile": {"tasks": ["marketing"]},
        "history": [
            {"user_id": "persona-seller", "item_id": "promo-guide", "type": "complete",
             "ts": "2026-10-01T09:00:00Z"},
        ],
    }  # fmt: skip
    paths["personas"].write_text(json.dumps([persona], ensure_ascii=False), encoding="utf-8")
    services = build_services(paths["catalog"], None, paths["personas"])
    with TestClient(create_app(services)) as client:
        assert client.get("/api/personas").json()[0]["id"] == "seller"
        result = recommend(client, profile={"tasks": ["marketing"]}, user_id="persona-seller")
        assert "promo-guide" not in ids(result["more"])


def test_page_static_files_and_docs(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/recommendations", "/api/events", "/api/meta", "/api/items/{item_id}"):
        assert path in paths
