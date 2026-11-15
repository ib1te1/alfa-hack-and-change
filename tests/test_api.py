import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alfarec.api.main import build_services, create_app
from alfarec.pipeline.build_index import prepare_texts
from alfarec.recsys.catalog import save_catalog
from alfarec.recsys.contracts import Item
from alfarec.recsys.retrieval import TfidfIndex, digest, texts_digest


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
def paths(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    # This fixture has an independent five-item catalog and a TF-IDF-only index.
    monkeypatch.setenv("ENCODER", "tfidf")
    catalog = tmp_path / "catalog.parquet"
    save_catalog(CATALOG, catalog)
    # The ranker checks that its index belongs to this exact catalog file.
    item_ids, texts = prepare_texts(CATALOG, source="metadata")
    TfidfIndex.fit(item_ids, texts).save(
        tmp_path / "index",
        text_source="metadata",
        catalog_sha256=digest(catalog),
        texts_sha256=texts_digest(item_ids, texts),
    )
    # Production weights belong to the production catalog, not this synthetic fixture.
    from alfarec.recsys.ranking import DEFAULT_WEIGHTS
    from alfarec.recsys.weights import weight_artifact

    weights_path = tmp_path / "initial-weights.json"
    weights_path.write_text(
        json.dumps(
            weight_artifact(
                DEFAULT_WEIGHTS,
                catalog,
                {"method": "initial weights for the synthetic API fixture"},
            )
        )
    )
    monkeypatch.setenv("WEIGHTS_PATH", str(weights_path))
    return {
        "catalog": catalog,
        "events": tmp_path / "events" / "events.jsonl",
        "personas": tmp_path / "personas.json",
        "index": tmp_path / "index",
    }


def services_for(paths: dict[str, Path]):
    return build_services(paths["catalog"], paths["events"], paths["personas"], paths["index"])


@pytest.fixture
def client(paths: dict[str, Path]):
    with TestClient(create_app(services_for(paths))) as test_client:
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
        "encoder": "tfidf",
        "recommender": "ranking",
    }


def test_saved_weights_change_real_http_scores(paths, monkeypatch):
    from alfarec.recsys.ranking import DEFAULT_WEIGHTS
    from alfarec.recsys.weights import weight_artifact

    monkeypatch.setenv("ENCODER", "tfidf")
    with TestClient(create_app(services_for(paths))) as client:
        before = recommend(client, profile={"tasks": ["marketing"]})
    weights = {key: 0.0 for key in DEFAULT_WEIGHTS} | {"task": 7.0}
    path = paths["catalog"].parent / "weights.json"
    path.write_text(json.dumps(weight_artifact(weights, paths["catalog"], {"method": "test"})))
    monkeypatch.setenv("WEIGHTS_PATH", str(path))
    with TestClient(create_app(services_for(paths))) as client:
        after = recommend(client, profile={"tasks": ["marketing"]})
        assert client.get("/api/health").json()["recommender"] == "ranking"

    def scores(answer):
        return {c["item"]["item_id"]: c["score"] for c in answer["more"]}

    assert scores(before) != scores(after)
    assert all(c["score"] in {0.0, 3.5} for c in after["more"])


def test_weight_overrides_and_explicit_disable(paths, monkeypatch):
    from alfarec.recsys.ranking import DEFAULT_WEIGHTS, load_ranker
    from alfarec.recsys.weights import weight_artifact

    path = paths["catalog"].parent / "weights.json"
    path.write_text(
        json.dumps(
            weight_artifact(DEFAULT_WEIGHTS | {"task": 7.0}, paths["catalog"], {"method": "test"})
        )
    )
    monkeypatch.setenv("WEIGHTS_PATH", str(path))
    loaded = load_ranker(paths["catalog"], paths["index"], weights={"task": 3.0})
    assert loaded._weights["task"] == 3.0
    # None must bypass even an explicitly configured, broken file for offline controls.
    path.write_text("{")
    control = load_ranker(paths["catalog"], paths["index"], weights_path=None)
    assert control._weights == DEFAULT_WEIGHTS


@pytest.mark.parametrize(
    "problem", ["missing", "corrupt", "stale", "nan", "unknown", "bool", "huge", "overflow"]
)
def test_invalid_weights_keep_api_available_with_explicit_fallback(paths, monkeypatch, problem):
    from alfarec.recsys.ranking import DEFAULT_WEIGHTS
    from alfarec.recsys.weights import weight_artifact

    path = paths["catalog"].parent / "weights.json"
    payload = weight_artifact(DEFAULT_WEIGHTS, paths["catalog"], {"method": "test"})
    if problem == "stale":
        payload["catalog_sha256"] = "0" * 64
    elif problem == "nan":
        payload["weights"]["task"] = float("nan")
    elif problem == "bool":
        payload["weights"]["task"] = True
    elif problem == "huge":
        payload["weights"]["task"] = 10**1000
    elif problem == "overflow":
        payload["weights"]["task"] = 1e308
    elif problem == "unknown":
        payload["weights"]["typo"] = 2.0
    if problem != "missing":
        path.write_text("{" if problem == "corrupt" else json.dumps(payload))
    monkeypatch.setenv("WEIGHTS_PATH", str(path))
    with TestClient(create_app(services_for(paths))) as client:
        health = client.get("/api/health").json()
        assert (health["status"], health["recommender"]) == ("degraded", "baseline")
        assert recommend(client, profile={"tasks": ["marketing"]})["more"]


def test_unusable_index_falls_back_to_the_baseline(paths: dict[str, Path]) -> None:
    services = build_services(
        paths["catalog"], paths["events"], paths["personas"], paths["index"].parent / "missing"
    )
    with TestClient(create_app(services)) as client:
        health = client.get("/api/health").json()
        assert (health["status"], health["recommender"], health["ready"]) == (
            "degraded", "baseline", True,
        )  # fmt: skip
        assert ids(recommend(client, profile={"tasks": ["marketing"]})["more"])


@pytest.mark.parametrize("index_problem", ["missing", "corrupt"])
@pytest.mark.parametrize("kind", ["complete", "known", "dislike"])
def test_fallback_keeps_feedback_exclusions(paths: dict[str, Path], index_problem, kind):
    if index_problem == "missing":
        paths["index"] = paths["index"].parent / "missing"
    else:
        (paths["index"] / "manifest.json").write_text("invalid", encoding="utf-8")
    with TestClient(create_app(services_for(paths))) as client:
        body = {"profile": {"tasks": ["marketing"]}, "user_id": "u1", "limit": 2}
        first = recommend(client, **body)
        item_id = first["more"][0]["item"]["item_id"]
        accepted = client.post(
            "/api/events",
            json={"user_id": "u1", "item_id": item_id, "type": kind,
                  "ts": "2026-11-14T10:00:00Z"},
        )  # fmt: skip
        assert accepted.status_code == 201
        after = recommend(client, **body)
        assert item_id not in ids(after["route"] + after["more"] + after["quick"])
        assert len(after["more"]) == 2
        assert item_id in ids(recommend(client, **{**body, "user_id": "u2"})["more"])
        assert item_id in ids(recommend(client, **body, mode="baseline")["more"])


def test_missing_catalog_degrades_with_a_clear_message(tmp_path: Path) -> None:
    services = build_services(
        tmp_path / "none.parquet", tmp_path / "e.jsonl", tmp_path / "none.json"
    )
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
    kinds = {"subscription": "subscribe", "tool": "calculate", "guide": "read"}
    result = recommend(client, profile={"tasks": ["marketing"], "time_budget": "course"})
    cards = result["route"] + result["more"] + result["quick"]
    assert {entry["item"]["access"] for entry in cards} >= {"free", "subscription"}
    for entry in cards:
        item, step = entry["item"], entry["next_step"]
        expected = kinds.get(item["access"]) or kinds.get(item["format"])
        if expected:
            assert step["kind"] == expected, item["item_id"]
        assert step["url"] == item["url"]


def test_feedback_events_change_the_next_answer_and_persist(
    client: TestClient, paths: dict[str, Path]
) -> None:
    body = {"profile": {"tasks": ["marketing"]}, "user_id": "u1"}
    first = recommend(client, **body)
    assert "promo-guide" in ids(first["route"] + first["more"] + first["quick"])
    event = {
        "user_id": "u1",
        "item_id": "promo-guide",
        "type": "known",
        "ts": "2026-11-14T10:00:00Z",
    }
    accepted = client.post("/api/events", json=event)
    assert accepted.status_code == 201
    assert accepted.json() == {"status": "accepted", "events_for_user": 1}
    after = recommend(client, **body)
    assert "promo-guide" not in ids(after["route"] + after["more"] + after["quick"])
    other = recommend(client, **{**body, "user_id": "u2"})
    assert "promo-guide" in ids(other["route"] + other["more"] + other["quick"])

    services = services_for(paths)
    assert [entry.item_id for entry in services.events.history("u1")] == ["promo-guide"]


def test_answers_carry_worded_reasons(client: TestClient) -> None:
    result = recommend(client, profile={"tasks": ["marketing"], "time_budget": "quick"})
    assert "promo-quick" in ids(result["route"] + result["quick"])
    for entry in result["route"] + result["more"] + result["quick"]:
        assert entry["reasons"] and all(reason["text"] for reason in entry["reasons"])
    assert result["profile_used"]["tasks"] == ["marketing"]


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
    with TestClient(create_app(services_for(paths))) as client:
        assert client.get("/api/personas").json()[0]["id"] == "seller"
        result = recommend(client, profile={"tasks": ["marketing"]}, user_id="persona-seller")
        assert "promo-guide" not in ids(result["route"] + result["more"] + result["quick"])


def test_page_static_files_and_docs(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/recommendations", "/api/events", "/api/meta", "/api/items/{item_id}"):
        assert path in paths


def test_platform_base_url_rewrites_links(paths: dict[str, Path], monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_BASE_URL", "https://stage.example.test/")
    with TestClient(create_app(services_for(paths))) as client:
        card = client.get("/api/items/plus-course").json()
        assert card["url"] == "https://stage.example.test/courses/plus"
        result = recommend(client, profile={"tasks": ["marketing"]}, limit=1)
        assert result["more"][0]["next_step"]["url"].startswith("https://stage.example.test/")
