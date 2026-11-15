import unicodedata
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import Field, StringConstraints, model_validator

from alfarec.config import load_configuration
from alfarec.recsys.contracts import Contract, ItemFormat, Level, Stage

TaxonId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Source dictionaries mix Latin look-alikes into Cyrillic titles ("Маркeтплейсы", "HoReСa").
# Both sides of a comparison are folded to Cyrillic, so either spelling yields the same key.
_HOMOGLYPHS = str.maketrans("aceopxyk", "асеорхук")


def match_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(text.translate(_HOMOGLYPHS).split())


class Taxon(Contract):
    label: Text
    description: Text


class SourceLinks(Contract):
    topics: list[Text] = Field(min_length=1)
    tags: list[Text] = Field(default_factory=list)
    article_subjects: list[Text] = Field(default_factory=list)


class TaskDefinition(Taxon):
    source_links: SourceLinks


class IndustryDefinition(Taxon):
    raw_values: list[Text] = Field(min_length=1)


class Taxonomy(Contract):
    version: Literal[1]
    source_links_status: Literal["pending_catalog_audit", "audited"]
    stages: dict[Stage, Taxon]
    levels: dict[Level, Taxon]
    tasks: Annotated[dict[TaxonId, TaskDefinition], Field(min_length=1)]
    industries: Annotated[dict[TaxonId, IndustryDefinition], Field(min_length=1)]
    formats: dict[ItemFormat, Taxon]

    @model_validator(mode="after")
    def validate_complete_vocabularies(self) -> Self:
        for name, expected in (("stages", Stage), ("levels", Level), ("formats", ItemFormat)):
            if set(getattr(self, name)) != set(get_args(expected)):
                raise ValueError(f"Taxonomy {name} must cover the complete contract vocabulary")
        return self


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    return load_configuration(path or Path(__file__).with_suffix(".yaml"), Taxonomy)
