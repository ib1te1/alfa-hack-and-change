from typing import Literal

from pydantic import Field

from alfarec.recsys.contracts import (
    Contract,
    Event,
    Identifier,
    Label,
    Mode,
    Profile,
)


class RecommendationRequest(Contract):
    profile: Profile = Field(default_factory=Profile)
    user_id: Identifier | None = None
    mode: Mode = "ours"
    limit: int = Field(default=10, ge=1, le=50, strict=True)


class HealthResponse(Contract):
    status: Literal["ok", "degraded"]
    ready: bool
    artifacts_loaded: bool
    catalog_items: int
    encoder: Literal["unavailable"] = "unavailable"
    recommender: Literal["interim", "unavailable"]


class MetaOption(Contract):
    id: str
    label: str
    description: str = ""


class MetaResponse(Contract):
    stages: list[MetaOption]
    levels: list[MetaOption]
    tasks: list[MetaOption]
    industries: list[MetaOption]
    formats: list[MetaOption]
    time_budgets: list[MetaOption]


class Persona(Contract):
    """A demo profile; its history is served under its own user_id."""

    id: Identifier
    title: Label
    legend: str = ""
    user_id: Identifier
    profile: Profile = Field(default_factory=Profile)
    history: list[Event] = Field(default_factory=list)


class EventAccepted(Contract):
    status: Literal["accepted"] = "accepted"
    events_for_user: int
