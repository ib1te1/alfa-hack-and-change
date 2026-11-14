from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alfarec.api.events import EventStore
from alfarec.api.main import DEFAULT_PERSONAS_PATH, build_services, create_app, load_personas
from alfarec.pipeline.build_catalog import DEFAULT_DUMP
from alfarec.pipeline.build_personas import build_personas, load_config, main, write_personas
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import Event
from alfarec.recsys.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(DEFAULT_CATALOG_PATH)


def bindings(catalog, users: dict[str, list[tuple[int, str]]]) -> dict:
    courses = [item.item_id for item in catalog if item.item_type == "course"]
    rows = []
    for user, entries in users.items():
        for index, status in entries:
            rows.append(
                {"UserId": user, "CourseId": courses[index], "Status": status, "IsDeleted": "f",
                 "Created": "2026-05-01 00:00:00+00", "Completed": None}
            )  # fmt: skip
    return {"CourseUserBindings": rows}


def test_committed_personas_cover_the_card(catalog):
    personas = load_personas(DEFAULT_PERSONAS_PATH)
    ids = {item.item_id for item in catalog}
    assert len(personas) == 8
    assert len({persona.id for persona in personas}) == len(personas)
    assert {"novice", "coffee", "seller", "services", "hiring", "outside"} <= {
        persona.id for persona in personas
    }
    assert sum(persona.id.startswith("real-") for persona in personas) == 2
    for persona in personas:
        assert all(event.item_id in ids for event in persona.history)
    novice = next(persona for persona in personas if persona.id == "novice")
    assert novice.history == [] and novice.profile.model_dump(exclude_defaults=True) == {}


def test_generation_is_seeded_and_hides_platform_user_ids(catalog, tmp_path: Path):
    tables = bindings(
        catalog,
        {
            "platform-user-a": [(0, "2"), (1, "1"), (2, "0")],
            "platform-user-b": [(3, "2"), (4, "2"), (5, "1")],
            "platform-user-c": [(6, "2")],
        },
    )
    config, taxonomy = load_config(), load_taxonomy()
    outputs = []
    for run in range(2):
        path = tmp_path / f"personas-{run}.json"
        write_personas(build_personas(config, catalog, taxonomy, tables), path)
        outputs.append(path.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]
    assert "platform-user" not in outputs[0]
    real = [
        persona
        for persona in load_personas(tmp_path / "personas-0.json")
        if persona.id.startswith("real-")
    ]
    assert len(real) == 2
    for persona in real:
        types = [event.type for event in persona.history]
        assert types.count("enroll") == 3 and "complete" in types
        assert types.index("enroll") < types.index("complete")


def test_history_changes_the_answer_and_stays_in_memory(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    services = build_services(DEFAULT_CATALOG_PATH, events, DEFAULT_PERSONAS_PATH)
    seller = next(persona for persona in services.personas if persona.id == "seller")
    completed = next(event.item_id for event in seller.history if event.type == "complete")
    with TestClient(create_app(services)) as client:
        body = {"profile": seller.profile.model_dump(exclude_defaults=True), "limit": 50}
        anonymous = client.post("/api/recommendations", json=body).json()
        personal = client.post("/api/recommendations", json={**body, "user_id": seller.user_id})
        assert completed in [entry["item"]["item_id"] for entry in anonymous["more"]]
        assert completed not in [entry["item"]["item_id"] for entry in personal.json()["more"]]
        feedback = {"user_id": seller.user_id, "item_id": completed, "type": "dislike",
                    "ts": "2026-11-14T10:00:00Z"}  # fmt: skip
        assert client.post("/api/events", json=feedback).status_code == 201
    assert not events.exists()


def test_event_store_ignores_stored_lines_of_protected_users(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    stored = Event(user_id="persona-x", item_id="a", type="known", ts="2026-11-01T00:00:00Z")
    path.write_text(stored.model_dump_json() + "\n", encoding="utf-8")
    assert EventStore(path, memory_only=["persona-x"]).history("persona-x") == []
    assert EventStore(path).history("persona-x") == [stored]


@pytest.mark.skipif(not DEFAULT_DUMP.is_file(), reason="the dump is handed over outside git")
def test_rebuild_from_dump_matches_committed_personas(tmp_path: Path):
    output = tmp_path / "personas.json"
    assert main(["--output", str(output)]) == 0
    assert output.read_bytes() == DEFAULT_PERSONAS_PATH.read_bytes()


def test_event_store_skips_a_truncated_line(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    good = Event(user_id="u", item_id="a", type="view", ts="2026-11-01T00:00:00Z")
    path.write_text(good.model_dump_json() + '\n{"user_id": "u", "item', encoding="utf-8")
    assert EventStore(path).history("u") == [good]


def test_invalid_platform_base_url_is_reported(monkeypatch):
    monkeypatch.setenv("PLATFORM_BASE_URL", "kurs.alfabank.ru")
    with pytest.raises(ValueError, match="PLATFORM_BASE_URL"):
        build_services(DEFAULT_CATALOG_PATH, None, DEFAULT_PERSONAS_PATH)
