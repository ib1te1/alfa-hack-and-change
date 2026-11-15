"""Freeze five public calibration pairs and compare independent human responses."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from alfarec.eval.profiles import ProfileSuite, load_profiles
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import (
    Access,
    Contract,
    Identifier,
    Item,
    ItemFormat,
    ItemType,
    Label,
    Profile,
)
from alfarec.recsys.taxonomy import Taxonomy, load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "calibration"
SELECTIONS = Path(__file__).with_name("calibration_pairs.json")
TEMPLATE = Path(__file__).with_name("calibration_form.html")
PROFILE_PATH = Path(__file__).with_name("profiles.yaml")
TAXONOMY_PATH = ROOT / "alfarec" / "recsys" / "taxonomy.yaml"
INSTRUCTIONS = ROOT / "docs" / "labeling.md"
Hash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PairId = Annotated[str, StringConstraints(pattern=r"^K0[1-5]$")]
ProfileId = Annotated[str, StringConstraints(pattern=r"^P\d{2}$")]
Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Score = Annotated[int, Field(strict=True, ge=0, le=2)] | Literal["unjudgeable"]
Version = Annotated[int, Field(strict=True, ge=1, le=1)]


class Selection(Contract):
    pair_id: PairId
    profile_id: ProfileId
    item_id: Identifier


def require_five_ids(values: list) -> None:
    if {value.pair_id for value in values} != {f"K{i:02}" for i in range(1, 6)}:
        raise ValueError("Exactly one answer or pair for each of K01..K05 is required")


class Selections(Contract):
    version: Version
    pairs: list[Selection] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_pairs(self) -> Self:
        require_five_ids(self.pairs)
        if (
            len({p.profile_id for p in self.pairs}) != 5
            or len({p.item_id for p in self.pairs}) != 5
        ):
            raise ValueError("Calibration requires five distinct profiles and materials")
        return self


class PublicCard(Contract):
    item_id: Identifier
    item_type: ItemType
    format: ItemFormat
    title: Label
    summary: str = Field(max_length=300)
    duration_min: int | None = Field(ge=1, strict=True)
    access: Access
    price_rub: int | None = Field(ge=0, strict=True)


class DisplayField(Contract):
    label: str
    value: str


class Pair(Contract):
    pair_id: PairId
    profile_id: ProfileId
    profile: Profile
    profile_fields: list[DisplayField]
    item: PublicCard
    format_label: str


class Sources(Contract):
    catalog: Hash
    profiles: Hash
    taxonomy: Hash
    instructions: Hash
    selections: Hash
    form_template: Hash


class Packet(Contract):
    version: Version
    sources: Sources
    pairs: list[Pair] = Field(min_length=5, max_length=5)
    packet_sha256: Hash

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        require_five_ids(self.pairs)
        if digest(self.model_dump(mode="json", exclude={"packet_sha256"})) != self.packet_sha256:
            raise ValueError("Calibration packet checksum mismatch")
        return self


class Answer(Contract):
    pair_id: PairId
    score: Score
    note: Note


class Response(Contract):
    version: Version
    packet_sha256: Hash
    rater_id: Literal["A", "B"]
    independent: Annotated[bool, Field(strict=True)]
    instruction_feedback: Note
    answers: list[Answer] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_answers(self) -> Self:
        require_five_ids(self.answers)
        if not self.independent:
            raise ValueError("The rater must confirm independent human assessment")
        return self


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path, *, binary: bool = False) -> str:
    # Text hashes survive Git's LF/CRLF conversion on Windows.
    data = path.read_bytes() if binary else path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> object:
    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON keys are not allowed")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant is not allowed: {value}")

    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique_keys,
        parse_constant=reject_constant,
    )


def profile_fields(profile: Profile, taxonomy: Taxonomy) -> list[DisplayField]:
    time_labels = {"quick": "До 5 минут", "hour": "До часа", "course": "Готов пройти курс"}
    values = [
        ("Стадия бизнеса", taxonomy.stages[profile.stage].label if profile.stage else "Не указана"),
        (
            "Сфера",
            ", ".join(taxonomy.industries[i].label for i in profile.industries) or "Не указана",
        ),
        ("Задача", ", ".join(taxonomy.tasks[t].label for t in profile.tasks) or "Не указана"),
        ("Опыт", taxonomy.levels[profile.experience].label if profile.experience else "Не указан"),
        ("Время", time_labels.get(profile.time_budget, "Не указано")),
        ("Запрос своими словами", profile.free_text or "Не указан"),
    ]
    return [DisplayField(label=label, value=value) for label, value in values]


def select_pairs(
    selections: Selections, suite: ProfileSuite, items: list[Item], taxonomy: Taxonomy
) -> list[Pair]:
    profiles = {p.id: p for p in suite.profiles}
    catalog = {item.item_id: item for item in items}
    pairs = []
    for selection in sorted(selections.pairs, key=lambda pair: pair.pair_id):
        profile = profiles[selection.profile_id]
        if profile.split != "tune" or profile.history:
            raise ValueError("Calibration uses only tune profiles without history selectors")
        item = catalog[selection.item_id]
        pairs.append(
            Pair(
                pair_id=selection.pair_id,
                profile_id=profile.id,
                profile=profile.profile,
                profile_fields=profile_fields(profile.profile, taxonomy),
                item=PublicCard.model_validate(
                    item.model_dump(include=PublicCard.model_fields.keys())
                ),
                format_label=taxonomy.formats[item.format].label,
            )
        )
    return pairs


def build_packet() -> Packet:
    taxonomy = load_taxonomy()
    selections = Selections.model_validate(read_json(SELECTIONS))
    sources = Sources(
        catalog=file_hash(DEFAULT_CATALOG_PATH, binary=True),
        profiles=file_hash(PROFILE_PATH),
        taxonomy=file_hash(TAXONOMY_PATH),
        instructions=file_hash(INSTRUCTIONS),
        selections=file_hash(SELECTIONS),
        form_template=file_hash(TEMPLATE),
    )
    pairs = select_pairs(selections, load_profiles(taxonomy=taxonomy), load_catalog(), taxonomy)
    payload = {
        "version": 1,
        "sources": sources.model_dump(),
        "pairs": [p.model_dump() for p in pairs],
    }
    return Packet.model_validate({**payload, "packet_sha256": digest(payload)})


def render_form(packet: Packet) -> str:
    # Even a title containing </script> must remain inert data in a local HTML file.
    data = canonical(packet.model_dump(mode="json")).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("__PACKET_JSON__", data)


def compare(packet: Packet, first: Response, second: Response) -> str:
    if first.rater_id == second.rater_id:
        raise ValueError("Two distinct human raters A and B are required")
    if any(r.packet_sha256 != packet.packet_sha256 for r in (first, second)):
        raise ValueError("Response belongs to a different calibration packet")
    raters = sorted((first, second), key=lambda response: response.rater_id)
    answers = [{answer.pair_id: answer for answer in r.answers} for r in raters]
    matches = sum(answers[0][p.pair_id].score == answers[1][p.pair_id].score for p in packet.pairs)
    lines = [
        "# B03: независимые оценки до обсуждения",
        "",
        f"Пакет: `{packet.packet_sha256}`.",
        f"Каталог: `{packet.sources.catalog}`.",
        "",
        f"Совпадений статусов: {matches} из 5 (включая unjudgeable).",
        "Это описательный итог калибровки инструкции, не метрика качества рекомендателя.",
        "Числовые оценки и unjudgeable не подменяются друг другом.",
    ]
    for rater in raters:
        count = sum(answer.score == "unjudgeable" for answer in rater.answers)
        lines += [
            "",
            f"Участник {rater.rater_id}: unjudgeable — {count} из 5.",
            "Замечание к инструкции:",
            quote(rater.instruction_feedback),
        ]
    for pair in packet.pairs:
        a, b = (values[pair.pair_id] for values in answers)
        lines += [
            "",
            f"## {pair.pair_id} · {pair.profile_id} · {pair.item.item_id}",
            "",
            quote(pair.item.title),
            "",
            f"A: **{a.score}**. Пояснение:",
            quote(a.note),
            "",
            f"B: **{b.score}**. Пояснение:",
            quote(b.note),
            "",
            "Статусы совпали." if a.score == b.score else "**Требуется разбор расхождения.**",
            "",
            "Итоговая оценка после обсуждения: не зафиксирована.",
            "Основание решения: не зафиксировано.",
        ]
    lines += [
        "",
        "## Решение по инструкции",
        "",
        "Не зафиксировано. Обсудить расхождения и неоднозначные случаи, сохранить исходные JSON,",
        "вписать итог по каждой паре, участников, дату и принятое изменение инструкции",
        "либо обоснование, почему изменение не требуется. До этого B03 остаётся review.",
        "",
    ]
    return "\n".join(lines)


def quote(value: str) -> str:
    # Human notes are text, not report markup, links, directives or HTML.
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in "\\`*_{}[]()!|":
        escaped = escaped.replace(char, "\\" + char)
    return "\n".join("> " + line for line in escaped.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Freeze the current pairs and offline form")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--check", action="store_true", help="Check without rewriting artifacts")
    comparison = commands.add_parser(
        "compare", help="Validate two human responses and draft a report"
    )
    comparison.add_argument("first", type=Path)
    comparison.add_argument("second", type=Path)
    comparison.add_argument("--packet", type=Path, default=DEFAULT_OUTPUT / "packet.json")
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        packet = build_packet()
        if args.command == "build":
            files = {
                "packet.json": packet.model_dump_json(indent=2) + "\n",
                "form.html": render_form(packet),
            }
            for name, contents in files.items():
                path = args.output_dir / name
                if args.check:
                    if path.read_text(encoding="utf-8") != contents:
                        raise ValueError(f"Stale calibration artifact: {path}")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(contents, encoding="utf-8", newline="\n")
            print(f"Calibration packet: {packet.packet_sha256}")
        else:
            saved = Packet.model_validate(read_json(args.packet))
            if saved != packet:
                raise ValueError("Packet is stale: rebuild and collect new independent ratings")
            responses = [Response.model_validate(read_json(p)) for p in (args.first, args.second)]
            report = compare(packet, *responses)
            # Preserve an existing report, including any human adjudication.
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(report)
            print(f"Comparison saved to {args.output}; human adjudication remains required")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Calibration error: {error}\n")


if __name__ == "__main__":
    main()
