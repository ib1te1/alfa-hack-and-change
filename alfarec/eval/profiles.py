from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from alfarec.config import ConfigurationError, load_configuration
from alfarec.recsys.contracts import Contract, EventType, ItemType, Level, Number, Profile
from alfarec.recsys.taxonomy import TaxonId, Taxonomy, Text, load_taxonomy


class HistorySelector(Contract):
    task: TaxonId
    item_type: ItemType
    level: Level | None = None


class HistoryStep(Contract):
    event: EventType
    selector: HistorySelector
    days_ago: int = Field(ge=0, strict=True)
    value: Number | None = None

    @model_validator(mode="after")
    def validate_rating(self) -> Self:
        if self.event == "rate" and (
            self.value is None or not 1 <= self.value <= 5 or not self.value.is_integer()
        ):
            raise ValueError("Rating history requires an integer score from 1 to 5")
        return self


class EvaluationProfile(Contract):
    id: Annotated[str, StringConstraints(pattern=r"^P\d{2}$")]
    audience: Literal["beginner_entrepreneur", "small_business_owner", "solo_founder"]
    case: Literal["standard", "history", "minimal", "out_of_scope"]
    split: Literal["tune", "holdout"]
    legend: Text
    profile: Profile
    history: list[HistoryStep] = Field(default_factory=list)
    expected_behavior: Literal["recommend", "cold_start", "clarify"]

    @model_validator(mode="after")
    def validate_scenario(self) -> Self:
        if (self.case == "history") != bool(self.history):
            raise ValueError("Only history cases must contain history selectors")
        if self.case == "out_of_scope" and (
            self.expected_behavior != "clarify" or not self.profile.free_text
        ):
            raise ValueError("Out-of-scope cases require a query and clarification")
        if self.case == "minimal" and self.profile.model_dump(exclude_defaults=True).keys() - {
            "free_text"
        }:
            raise ValueError("Minimal profiles contain only free text or no information")
        return self


class ProfileSuite(Contract):
    version: Literal[1]
    profiles: Annotated[list[EvaluationProfile], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        if len({profile.id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("Evaluation profile IDs must be unique")
        return self

    def validate_taxonomy(self, taxonomy: Taxonomy) -> None:
        for case in self.profiles:
            tasks = set(case.profile.tasks) | {step.selector.task for step in case.history}
            unknown_tasks = tasks - taxonomy.tasks.keys()
            unknown_industries = set(case.profile.industries) - taxonomy.industries.keys()
            if unknown_tasks or unknown_industries:
                raise ValueError(
                    f"{case.id}: unknown tasks {sorted(unknown_tasks)} "
                    f"or industries {sorted(unknown_industries)}"
                )


def load_profiles(path: Path | None = None, taxonomy: Taxonomy | None = None) -> ProfileSuite:
    source = path or Path(__file__).with_suffix(".yaml")
    suite = load_configuration(source, ProfileSuite)
    try:
        suite.validate_taxonomy(taxonomy if taxonomy is not None else load_taxonomy())
    except ValueError as error:
        raise ConfigurationError(f"Invalid configuration at {source}: {error}") from error
    return suite
