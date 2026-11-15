from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Number = Annotated[FiniteFloat, Field(strict=True)]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ProfileLabels = Annotated[list[Label], Field(max_length=20)]
Stage = Literal["idea", "launch", "operating", "growth"]
Level = Literal["beginner", "intermediate", "advanced"]
TimeBudget = Literal["quick", "hour", "course"]
Mode = Literal["ours", "baseline"]
ItemType = Literal["article", "document", "course", "webinar", "podcast", "glossary", "calculator"]
ItemFormat = Literal["quick_answer", "guide", "template", "tool", "course", "recording", "podcast"]
Access = Literal["free", "subscription", "paid"]
EventType = Literal["view", "complete", "save", "enroll", "rate", "known", "dislike"]
NextStepKind = Literal[
    "read", "download", "calculate", "start_course", "watch", "listen", "subscribe", "purchase"
]


def parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("Timestamp must be an ISO 8601 string or an aware datetime")


Timestamp = Annotated[AwareDatetime, BeforeValidator(parse_timestamp)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Profile(Contract):
    stage: Stage | None = None
    industries: ProfileLabels = Field(default_factory=list)
    tasks: ProfileLabels = Field(default_factory=list)
    experience: Level | None = None
    time_budget: TimeBudget | None = None
    free_text: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] | None = (
        None
    )

    @field_validator(
        "industries", "tasks", mode="before", json_schema_input_type=ProfileLabels | None
    )
    @classmethod
    def normalize_missing_lists(cls, value: object) -> object:
        return [] if value is None else value


class Item(Contract):
    item_id: Identifier
    item_type: ItemType
    format: ItemFormat
    title: Label
    summary: str = Field(default="", max_length=300)
    duration_min: int | None = Field(default=None, ge=1, strict=True)
    access: Access
    price_rub: int | None = Field(default=None, ge=0, strict=True)
    url: HttpUrl
    topics: list[Label] = Field(default_factory=list)
    raw_tags: list[Label] = Field(default_factory=list)
    raw_industries: list[Label] = Field(default_factory=list)
    subject: Label | None = None
    published_at: Timestamp | None = None
    popularity: Number | None = Field(default=None, ge=0)
    completion_rate: Number | None = Field(default=None, ge=0, le=1)
    rating: Number | None = Field(default=None, ge=1, le=5)


class Event(Contract):
    user_id: Identifier
    item_id: Identifier
    type: EventType
    value: Number | None = None
    ts: Timestamp

    @model_validator(mode="after")
    def validate_rating(self) -> Self:
        if self.type == "rate" and (
            self.value is None or not 1 <= self.value <= 5 or not self.value.is_integer()
        ):
            raise ValueError("A rate event requires an integer value from 1 to 5")
        return self


class ReasonCode(StrEnum):
    TASK_MATCH = "task_match"
    STAGE_FIT = "stage_fit"
    BEGINNER_FRIENDLY = "beginner_friendly"
    INDUSTRY_MATCH = "industry_match"
    CONTINUES_HISTORY = "continues_history"
    POPULAR_WITH_SIMILAR = "popular_with_similar"
    SHORT_FORMAT = "short_format"
    FRESH = "fresh"
    NEXT_LEVEL = "next_level"
    DEEP_DIVE_COURSE = "deep_dive_course"


class Reason(Contract):
    code: ReasonCode
    weight: Number
    params: dict[str, str | Number | bool | None] = Field(default_factory=dict)
    # Filled by explain.py from code and params; rankers leave it empty.
    text: Label | None = None


class NextStep(Contract):
    kind: NextStepKind
    label: Label
    url: HttpUrl


class Recommendation(Contract):
    item: Item
    score: Number
    reasons: list[Reason] = Field(default_factory=list)
    next_step: NextStep


class Clarification(Contract):
    question: Label
    options: list[Label] = Field(min_length=2, max_length=4)


class RecommendationResponse(Contract):
    route: list[Recommendation] = Field(default_factory=list)
    more: list[Recommendation] = Field(default_factory=list)
    quick: list[Recommendation] = Field(default_factory=list)
    clarification: Clarification | None = None
    # One line above the route on why the steps go in this order; set by explain.py.
    route_note: Annotated[str, StringConstraints(max_length=300)] | None = None
    mode: Mode
    profile_used: Profile


class Recommender(Protocol):
    def recommend(
        self,
        profile: Profile,
        history: Sequence[Event],
        mode: Mode = "ours",
        limit: int = 10,
    ) -> RecommendationResponse: ...
