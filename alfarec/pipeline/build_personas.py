"""Build demo personas with history for the page and /api/personas.

Usage: python -m alfarec.pipeline.build_personas [--dump PATH] [--output PATH]
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field

from alfarec.api.schemas import Persona
from alfarec.config import load_configuration
from alfarec.pipeline.build_catalog import DEFAULT_DUMP, _parse_timestamp
from alfarec.pipeline.dump_reader import Tables, read_tables
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import Contract, Event, Item, Profile, Timestamp
from alfarec.recsys.taxonomy import Taxonomy, load_taxonomy, match_key

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("personas.yaml")
DEFAULT_OUTPUT = ROOT / "artifacts" / "personas.json"
VIEW_TYPES = ("article", "document")
COMPLETED_STATUS = "2"


class HistoryEntry(Contract):
    event: Literal["view", "complete", "save", "enroll", "known", "dislike"]
    title: str
    days_ago: int = Field(ge=0)


class SyntheticViews(Contract):
    # One task keeps the views on topic: broad topics of a second task would dilute them.
    task: str
    count: int = Field(ge=1, le=20)
    days: int = Field(ge=1)


class PersonaSpec(Contract):
    id: str
    title: str
    legend: str
    profile: Profile = Field(default_factory=Profile)
    history: list[HistoryEntry] = Field(default_factory=list)
    synthetic_views: SyntheticViews | None = None


class RealUsers(Contract):
    count: int = Field(ge=0)
    min_courses: int = Field(ge=2)


class PersonasConfig(Contract):
    version: Literal[1]
    anchor_date: Timestamp
    seed: int
    personas: list[PersonaSpec]
    real_users: RealUsers


def _find_by_title(catalog: Sequence[Item], title: str) -> Item:
    key = match_key(title)
    found = [item for item in catalog if match_key(item.title) == key]
    if len(found) != 1:
        raise ValueError(f"Expected one catalog item titled {title!r}, found {len(found)}")
    return found[0]


def _on_task(item: Item, task_id: str, taxonomy: Taxonomy) -> bool:
    links = taxonomy.tasks[task_id].source_links
    topics = {match_key(topic) for topic in links.topics}
    subjects = {match_key(subject) for subject in links.article_subjects}
    if item.subject and item.item_type == "article" and match_key(item.subject) in subjects:
        return True
    return bool(topics & {match_key(topic) for topic in item.topics})


def _synthetic_views(
    spec: PersonaSpec,
    config: PersonasConfig,
    catalog: Sequence[Item],
    taxonomy: Taxonomy,
    user_id: str,
    taken: set[str],
) -> list[Event]:
    views = spec.synthetic_views
    if views is None:
        return []
    if views.task not in taxonomy.tasks:
        raise ValueError(f"Persona {spec.id} refers to unknown task {views.task}")
    pool = sorted(
        item.item_id
        for item in catalog
        if item.item_type in VIEW_TYPES
        and item.access == "free"
        and item.item_id not in taken
        and _on_task(item, views.task, taxonomy)
    )
    # The seed is mixed with the persona id so adding a persona does not reshuffle the others.
    rng = random.Random(f"{config.seed}:{spec.id}")
    chosen = rng.sample(pool, min(views.count, len(pool)))
    return [
        Event(
            user_id=user_id,
            item_id=item_id,
            type="view",
            ts=config.anchor_date - timedelta(days=rng.randint(0, views.days - 1), minutes=index),
        )
        for index, item_id in enumerate(chosen)
    ]


def scripted_personas(
    config: PersonasConfig, catalog: Sequence[Item], taxonomy: Taxonomy
) -> list[Persona]:
    personas = []
    for spec in config.personas:
        user_id = f"persona-{spec.id}"
        history = [
            Event(
                user_id=user_id,
                item_id=_find_by_title(catalog, entry.title).item_id,
                type=entry.event,
                ts=config.anchor_date - timedelta(days=entry.days_ago),
            )
            for entry in spec.history
        ]
        taken = {event.item_id for event in history}
        history += _synthetic_views(spec, config, catalog, taxonomy, user_id, taken)
        personas.append(
            Persona(
                id=spec.id,
                title=spec.title,
                legend=spec.legend,
                user_id=user_id,
                profile=spec.profile,
                history=sorted(history, key=lambda event: (event.ts, event.item_id)),
            )
        )
    return personas


def real_personas(config: PersonasConfig, catalog: Sequence[Item], tables: Tables) -> list[Persona]:
    """Users with several courses; their platform ids never leave this function."""
    courses = {item.item_id: item for item in catalog if item.item_type == "course"}
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in tables["CourseUserBindings"]:
        if row["IsDeleted"] != "t" and row["CourseId"] in courses:
            by_user[row["UserId"]].append(row)
    candidates = [
        user
        for user, rows in by_user.items()
        if len({row["CourseId"] for row in rows}) >= config.real_users.min_courses
        and any(row["Status"] == COMPLETED_STATUS for row in rows)
    ]
    candidates.sort(key=lambda user: hashlib.sha256(f"{config.seed}:{user}".encode()).hexdigest())
    personas = []
    for number, user in enumerate(candidates[: config.real_users.count], start=1):
        user_id = f"real-{number}"
        history = []
        for row in by_user[user]:
            started = _parse_timestamp(row["Created"]) or config.anchor_date
            history.append(
                Event(user_id=user_id, item_id=row["CourseId"], type="enroll", ts=started)
            )
            if row["Status"] == COMPLETED_STATUS:
                finished = _parse_timestamp(row["Completed"]) or started
                history.append(
                    Event(user_id=user_id, item_id=row["CourseId"], type="complete", ts=finished)
                )
        history.sort(key=lambda event: (event.ts, event.type != "enroll", event.item_id))
        titles = sorted({courses[event.item_id].title for event in history})
        personas.append(
            Persona(
                id=user_id,
                title=f"Реальный пользователь {number}",
                legend=(
                    f"Обезличенная история платформы, курсов: {len(titles)} — "
                    + "; ".join(titles)
                    + ". Профиль не заполнен: в данных его нет."
                ),
                user_id=user_id,
                history=history,
            )
        )
    return personas


def build_personas(
    config: PersonasConfig, catalog: Sequence[Item], taxonomy: Taxonomy, tables: Tables
) -> list[Persona]:
    return [
        *scripted_personas(config, catalog, taxonomy),
        *real_personas(config, catalog, tables),
    ]


def write_personas(personas: Sequence[Persona], path: Path) -> None:
    payload = [persona.model_dump(mode="json", exclude_defaults=True) for persona in personas]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_config(path: Path = SPEC_PATH) -> PersonasConfig:
    return load_configuration(path, PersonasConfig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.dump.is_file():
        print(f"Dump not found: {args.dump}", file=sys.stderr)
        return 1
    personas = build_personas(
        load_config(),
        load_catalog(args.catalog),
        load_taxonomy(),
        read_tables(args.dump, ["CourseUserBindings"]),
    )
    write_personas(personas, args.output)
    print(f"{len(personas)} personas: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
