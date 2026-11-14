import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from alfarec.api.schemas import RecommendationRequest
from alfarec.recsys.contracts import (
    Clarification,
    Event,
    Item,
    NextStep,
    Profile,
    Reason,
    Recommendation,
    RecommendationResponse,
)


def test_empty_and_null_profile_support_cold_start() -> None:
    assert Profile() == Profile.model_validate(
        dict.fromkeys(["stage", "industries", "tasks", "experience", "time_budget", "free_text"])
    )
    first, second = Profile(), Profile()
    first.tasks.append("marketing")
    assert second.tasks == []
    assert RecommendationRequest().profile == Profile()


@pytest.mark.parametrize(
    "payload",
    [
        {"stage": "unknown"},
        {"experience": "expert"},
        {"time_budget": "tomorrow"},
        {"tasks": [""]},
        {"industries": [" "]},
        {"free_text": "x" * 2001},
        {"hidden_field": "value"},
    ],
)
def test_profile_rejects_invalid_inputs(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Profile.model_validate(payload)


@pytest.mark.parametrize("limit", [0, -1, 51, 1.5, True, "10"])
def test_request_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(limit=limit)


@pytest.mark.parametrize("value", [None, 0, 6, 2.5, True, "4", float("nan"), float("inf")])
def test_rating_requires_valid_score(value: object) -> None:
    with pytest.raises(ValidationError):
        Event(user_id="demo", item_id="item-1", type="rate", value=value, ts=datetime.now(UTC))


def test_event_requires_aware_time_and_preserves_it_in_json() -> None:
    event = Event.model_validate(
        {
            "user_id": "demo",
            "item_id": "item-1",
            "type": "rate",
            "value": 4,
            "ts": "2026-11-14T10:00:00+04:00",
        }
    )
    assert Event.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError):
        Event(user_id="demo", item_id="item-1", type="view", ts="2026-11-14T10:00:00")


@pytest.mark.parametrize("timestamp", [10, True, "10", "2026-11-14"])
def test_events_reject_numeric_or_naive_timestamps(timestamp: object) -> None:
    with pytest.raises(ValidationError):
        Event(user_id="demo", item_id="item-1", type="view", ts=timestamp)


def test_profile_schema_distinguishes_nullable_input_from_normalized_output() -> None:
    input_schema = RecommendationRequest.model_json_schema()["$defs"]["Profile"]["properties"]
    output_schema = Profile.model_json_schema(mode="serialization")["properties"]
    for field in ("tasks", "industries"):
        assert {variant["type"] for variant in input_schema[field]["anyOf"]} == {"array", "null"}
        array_schema = next(s for s in input_schema[field]["anyOf"] if s["type"] == "array")
        assert array_schema["maxItems"] == 20
        assert output_schema[field]["type"] == "array"


@pytest.mark.parametrize("field", ["index_text", "content", "lessons", "LessonEmbedItems"])
def test_public_item_rejects_content_fields(catalog: list[Item], field: str) -> None:
    payload = catalog[0].model_dump()
    payload[field] = "private text"
    with pytest.raises(ValidationError):
        Item.model_validate(payload)


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/item", "/relative/path"])
def test_links_require_http_urls(catalog: list[Item], url: str) -> None:
    with pytest.raises(ValidationError):
        Item.model_validate({**catalog[0].model_dump(), "url": url})
    with pytest.raises(ValidationError):
        NextStep(kind="read", label="Читать", url=url)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ranking_numbers_must_be_finite(catalog: list[Item], value: float) -> None:
    with pytest.raises(ValidationError):
        Reason(code="task_match", weight=value)
    with pytest.raises(ValidationError):
        Recommendation(
            item=catalog[0],
            score=value,
            next_step=NextStep(kind="read", label="Читать", url=catalog[0].url),
        )


def test_unknown_reason_and_event_types_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Reason(code="invented", weight=1)
    with pytest.raises(ValidationError):
        Event(user_id="demo", item_id="item-1", type="invented", ts=datetime.now(UTC))


def test_response_round_trip_preserves_explanation(catalog: list[Item]) -> None:
    response = RecommendationResponse(
        route=[
            Recommendation(
                item=catalog[0],
                score=0.8,
                reasons=[Reason(code="task_match", weight=0.4, params={"task": "marketing"})],
                next_step=NextStep(kind="read", label="Читать", url=catalog[0].url),
            )
        ],
        clarification=Clarification(question="Что ближе?", options=["Маркетинг", "Продажи"]),
        mode="ours",
        profile_used=Profile(tasks=["marketing"]),
    )
    encoded = response.model_dump_json()
    json.loads(encoded)
    assert RecommendationResponse.model_validate_json(encoded) == response
