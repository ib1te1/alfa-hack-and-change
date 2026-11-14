"""Build the public catalog and the embedding text from the platform dump.

Usage: python -m alfarec.pipeline.build_catalog [--dump PATH] [--base-url URL]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean

import pyarrow as pa

from alfarec.pipeline.dump_reader import Row, Tables, read_tables
from alfarec.pipeline.text import html_to_text, normalize_label, parse_duration, truncate
from alfarec.recsys.catalog import save_catalog, write_parquet
from alfarec.recsys.contracts import Item

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUMP = ROOT / "data" / "alfa_course_hackathon.backup"
DEFAULT_CATALOG = ROOT / "artifacts" / "catalog.parquet"
DEFAULT_INDEX_TEXT = ROOT / "data" / "processed" / "index_text.parquet"
# Links inside the dump's own articles point to this host with the same slugs.
DEFAULT_BASE_URL = "https://kurs.alfabank.ru"

PUBLISHED = "Опубликовано"
SUMMARY_LIMIT = 300
TITLE_LIMIT = 200
BODY_LIMIT = 4000
QUICK_ANSWER_MAX_MIN = 5
COMPLETED_STATUS = "2"

INDEX_TEXT_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("item_type", pa.string(), nullable=False),
        pa.field("access", pa.string(), nullable=False),
        pa.field("index_text", pa.string(), nullable=False),
    ]
)

TABLES = (
    "Statuses", "Articles", "ArticleSubjects", "ArticleEmbedItems", "Documents", "DocumentTypes",
    "DocumentEmbedItems", "Courses", "CourseTypes", "CourseSubscriptions", "CourseInfoBindings",
    "CourseLessonBindings", "CourseUserBindings", "Landings", "Webinars", "Podcasts",
    "PodcastDetails", "PodcastSubscriptions", "Glossary", "GlossaryEmbedItems", "CalculatorPages",
    "CalculatorPageEmbedItems", "ContentSettings", "ContentSettingsTopics", "ContentSettingsTags",
    "ContentSettingsIndustries", "Topics", "Tags", "Industries", "Subscription",
    "SubscriptionTariffs", "UserRatings",
)  # fmt: skip

# Path segment of the public page for each material type on the platform.
URL_PATHS = {
    "article": "articles",
    "document": "documents",
    "course": "courses",
    "webinar": "webinars",
    "podcast": "podcasts",
    "glossary": "vocabulary",
    "calculator": "calculators",
}


@dataclass
class Draft:
    item_id: str
    item_type: str
    title: str
    description: str
    slug: str
    published: str | None
    duration: str | None = None
    access: str = "free"
    price_rub: int | None = None
    subject: str | None = None
    # Public text in addition to title and description; never lesson content.
    extra_text: list[str] = field(default_factory=list)
    body: str = ""


def _true(value: str | None) -> bool:
    return value == "t"


def _live(rows: Iterable[Row], published_id: str | None = None) -> list[Row]:
    """Rows that are not deleted and, when the table has a status, published."""
    result = []
    for row in rows:
        if _true(row.get("IsDeleted")):
            continue
        if published_id is not None and row.get("StatusId") != published_id:
            continue
        result.append(row)
    return result


def _by_id(rows: Iterable[Row]) -> dict[str, Row]:
    return {row["Id"]: row for row in rows}


def _ordered(rows: Iterable[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: (int(row.get("Order") or 0), row.get("Id") or ""))


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    # pg_dump writes offsets as "+00"; Python wants hours and minutes.
    if len(text) > 3 and text[-3] in "+-" and text[-3:].lstrip("+-").isdigit():
        text += ":00"
    return datetime.fromisoformat(text)


def _embed_text(rows: Iterable[Row], owner_column: str) -> dict[str, str]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row[owner_column]].append(row)
    return {
        owner: "\n".join(
            filter(None, (html_to_text(row["HtmlContent"]) for row in _ordered(items)))
        )
        for owner, items in grouped.items()
    }


def _landing_text(landing: Row) -> list[str]:
    parts = [
        html_to_text(landing.get("Description")),
        html_to_text(landing.get("LandingDescription")),
    ]
    try:
        bindings = json.loads(landing.get("InfoBindings") or "[]")
    except json.JSONDecodeError:
        bindings = []
    for binding in sorted(bindings, key=lambda entry: entry.get("Order") or 0):
        parts.append(html_to_text(binding.get("Title")))
        parts.append(html_to_text(binding.get("Description")))
    return parts


class CatalogBuilder:
    def __init__(self, tables: Tables, base_url: str) -> None:
        self.tables = tables
        self.base_url = base_url.rstrip("/")
        statuses = {row["Title"]: row["Id"] for row in tables["Statuses"]}
        self.published_id = statuses.get(PUBLISHED)
        if self.published_id is None:
            raise ValueError(f"Status {PUBLISHED!r} is missing from the dump")

    def live(self, name: str, *, with_status: bool = True) -> list[Row]:
        return _live(self.tables[name], self.published_id if with_status else None)

    def drafts(self) -> list[Draft]:
        return [
            *self.articles(),
            *self.documents(),
            *self.courses(),
            *self.webinars(),
            *self.podcasts(),
            *self.glossary(),
            *self.calculators(),
        ]

    def articles(self) -> list[Draft]:
        subjects = _by_id(self.live("ArticleSubjects", with_status=False))
        bodies = _embed_text(self.tables["ArticleEmbedItems"], "ArticleId")
        subscription_price = self.subscription_price()
        drafts = []
        for row in self.live("Articles"):
            subject = subjects.get(row["SubjectId"] or "")
            closed = _true(row["IsClosed"])
            drafts.append(
                Draft(
                    item_id=row["Id"],
                    item_type="article",
                    title=row["Title"],
                    description=row["Description"] or row["CardDescription"] or "",
                    slug=row["Url"],
                    published=row["Published"],
                    duration=row["Duration"],
                    # Closed articles are readable only with the "Курс+" subscription.
                    access="subscription" if closed else "free",
                    price_rub=subscription_price if closed else None,
                    subject=normalize_label(subject["Title"]) if subject else None,
                    extra_text=[html_to_text(row["ContentHeadings"])],
                    body="" if closed else bodies.get(row["Id"], ""),
                )
            )
        return drafts

    def documents(self) -> list[Draft]:
        types = _by_id(self.live("DocumentTypes", with_status=False))
        bodies = _embed_text(self.tables["DocumentEmbedItems"], "DocumentId")
        drafts = []
        for row in self.live("Documents"):
            document_type = types.get(row["TypeId"] or "")
            drafts.append(
                Draft(
                    item_id=row["Id"],
                    item_type="document",
                    title=row["Title"],
                    description=row["Description"] or "",
                    slug=row["Url"],
                    published=row["Published"],
                    subject=normalize_label(document_type["Title"]) if document_type else None,
                    body=bodies.get(row["Id"], ""),
                )
            )
        return drafts

    def subscription_price(self) -> int | None:
        """Monthly price of the main "Курс+" tariff, discounted if a discount is set."""
        subscriptions = self.live("Subscription", with_status=False)
        subscription_ids = {row["Id"] for row in subscriptions}
        tariffs = [
            row for row in self.tables["SubscriptionTariffs"]
            if row["SubscriptionId"] in subscription_ids and _true(row["IsMain"])
        ]  # fmt: skip
        if not tariffs:
            return None
        tariff = tariffs[0]
        return int(tariff["DiscountCost"] or tariff["Cost"])

    def courses(self) -> list[Draft]:
        course_types = {row["Id"]: row["Title"] for row in self.tables["CourseTypes"]}
        subscribed = {row["CourseId"] for row in self.tables["CourseSubscriptions"]}
        landings = {row["Url"]: row for row in self.live("Landings")}
        info: dict[str, list[Row]] = defaultdict(list)
        for row in self.tables["CourseInfoBindings"]:
            info[row["CourseId"]].append(row)
        subscription_price = self.subscription_price()
        drafts = []
        for row in self.live("Courses"):
            cost = int(row["Cost"]) if row["Cost"] else None
            course_type = course_types.get(row["TypeId"] or "")
            if cost or course_type == "Платный":
                access, price = "paid", cost
            elif row["Id"] in subscribed or course_type == "По подписке":
                access, price = "subscription", subscription_price
            else:
                access, price = "free", None
            extra = [html_to_text(row["DescriptionAdditional"]), html_to_text(row["Target"])]
            for binding in _ordered(info.get(row["Id"], [])):
                extra += [html_to_text(binding["Title"]), html_to_text(binding["Description"])]
            landing = landings.get(row["Url"])
            if landing:
                extra += _landing_text(landing)
            drafts.append(
                Draft(
                    item_id=row["Id"],
                    item_type="course",
                    title=row["Title"],
                    description=row["Description"],
                    slug=row["Url"],
                    published=row["Published"],
                    duration=row["Duration"],
                    access=access,
                    price_rub=price,
                    extra_text=extra,
                )
            )
        return drafts

    def webinars(self) -> list[Draft]:
        return [
            Draft(
                item_id=row["Id"],
                item_type="webinar",
                title=row["Title"],
                description=row["Description"] or "",
                slug=row["Url"],
                published=row["StartsAt"] or row.get("Published"),
                duration=row["Duration"],
                extra_text=[html_to_text(row["DescriptionExt"])],
            )
            for row in self.live("Webinars")
        ]

    def podcasts(self) -> list[Draft]:
        subscribed = {row["PodcastId"] for row in self.tables["PodcastSubscriptions"]}
        subscription_price = self.subscription_price()
        details: dict[str, list[Row]] = defaultdict(list)
        for row in self.tables["PodcastDetails"]:
            details[row["PodcastId"]].append(row)
        drafts = []
        for row in self.live("Podcasts"):
            cost = int(row["Cost"]) if row["Cost"] else None
            if cost:
                access, price = "paid", cost
            elif row["Id"] in subscribed:
                access, price = "subscription", subscription_price
            else:
                access, price = "free", None
            extra = [html_to_text(row["DescriptionExt"])]
            for detail in _ordered(details.get(row["Id"], [])):
                extra += [html_to_text(detail["Title"]), html_to_text(detail["Description"])]
            drafts.append(
                Draft(
                    item_id=row["Id"],
                    item_type="podcast",
                    title=row["Title"],
                    description=row["Description"],
                    slug=row["Url"],
                    published=row["Published"],
                    duration=row["Duration"],
                    access=access,
                    price_rub=price,
                    extra_text=extra,
                )
            )
        return drafts

    def glossary(self) -> list[Draft]:
        bodies = _embed_text(self.tables["GlossaryEmbedItems"], "GlossaryId")
        return [
            Draft(
                item_id=row["Id"],
                item_type="glossary",
                title=row["Title"],
                description=row["Description"] or "",
                slug=row["Url"],
                published=row["Published"],
                duration=row["Duration"],
                body=bodies.get(row["Id"], ""),
            )
            for row in self.live("Glossary")
        ]

    def calculators(self) -> list[Draft]:
        # Ratings reference calculator pages, so the page is the catalog item.
        bodies = _embed_text(self.tables["CalculatorPageEmbedItems"], "CalculatorPageId")
        return [
            Draft(
                item_id=row["Id"],
                item_type="calculator",
                title=row["Title"],
                description=row["Description"] or "",
                slug=row["Url"],
                published=row["Published"],
                body=bodies.get(row["Id"], ""),
            )
            for row in self.live("CalculatorPages")
        ]

    def labels(self) -> dict[str, dict[str, list[str]]]:
        """Topics, tags and industries per material id, from the polymorphic ContentSettings."""
        settings = {
            row["Id"]: row["EntityId"] for row in self.live("ContentSettings", with_status=False)
        }
        dictionaries = {
            "topics": ("ContentSettingsTopics", "TopicId", "Topics"),
            "raw_tags": ("ContentSettingsTags", "TagId", "Tags"),
            "raw_industries": ("ContentSettingsIndustries", "IndustryId", "Industries"),
        }
        result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, (link_table, ref_column, dictionary) in dictionaries.items():
            titles = {
                row["Id"]: normalize_label(row["Title"])
                for row in self.live(dictionary, with_status=False)
            }
            for link in _ordered(self.tables[link_table]):
                entity = settings.get(link["ContentSettingsId"])
                title = titles.get(link[ref_column])
                if entity and title:
                    result[entity][key].append(title)
        return {
            entity: {key: _unique(values) for key, values in labels.items()}
            for entity, labels in result.items()
        }

    def engagement(self) -> dict[str, dict[str, float]]:
        """Popularity, completion and rating, available only for courses and calculators."""
        stats: dict[str, dict[str, float]] = {}
        enrollments: dict[str, list[str]] = defaultdict(list)
        for row in _live(self.tables["CourseUserBindings"]):
            enrollments[row["CourseId"]].append(row["Status"])
        for course_id, statuses in enrollments.items():
            stats[course_id] = {
                "popularity": float(len(statuses)),
                "completion_rate": round(statuses.count(COMPLETED_STATUS) / len(statuses), 4),
            }
        lesson_course = {
            row["LessonId"]: row["CourseId"] for row in _live(self.tables["CourseLessonBindings"])
        }
        calculator_ids = {row["Id"] for row in self.tables["CalculatorPages"]}
        ratings: dict[str, list[int]] = defaultdict(list)
        for row in _live(self.tables["UserRatings"]):
            value = int(row["Value"])
            if not 1 <= value <= 5:
                continue
            entity = row["EntityId"]
            if entity in calculator_ids:
                ratings[entity].append(value)
            elif entity in lesson_course:
                ratings[lesson_course[entity]].append(value)
        for entity, values in ratings.items():
            entry = stats.setdefault(entity, {})
            entry["rating"] = round(fmean(values), 4)
            if entity in calculator_ids:
                # Calculators have no visit log; the number of ratings stands in for usage.
                entry["popularity"] = float(len(values))
        return stats

    def build(self) -> tuple[list[Item], list[dict[str, str]]]:
        labels = self.labels()
        engagement = self.engagement()
        items: list[Item] = []
        index_rows: list[dict[str, str]] = []
        for draft in sorted(self.drafts(), key=lambda entry: (entry.item_type, entry.item_id)):
            title = truncate(html_to_text(draft.title), TITLE_LIMIT)
            description = html_to_text(draft.description)
            duration = parse_duration(draft.duration)
            item_format = _format(draft.item_type, duration)
            item_labels = labels.get(draft.item_id, {})
            stats = engagement.get(draft.item_id, {})
            item = Item.model_validate(
                {
                    "item_id": draft.item_id,
                    "item_type": draft.item_type,
                    "format": item_format,
                    "title": title,
                    "summary": truncate(description, SUMMARY_LIMIT),
                    "duration_min": duration,
                    "access": draft.access,
                    "price_rub": draft.price_rub,
                    "url": f"{self.base_url}/{URL_PATHS[draft.item_type]}/{draft.slug.strip('/')}",
                    "topics": item_labels.get("topics", []),
                    "raw_tags": item_labels.get("raw_tags", []),
                    "raw_industries": item_labels.get("raw_industries", []),
                    "subject": draft.subject or None,
                    "published_at": _parse_timestamp(draft.published),
                    "popularity": stats.get("popularity"),
                    "completion_rate": stats.get("completion_rate"),
                    "rating": stats.get("rating"),
                }
            )
            items.append(item)
            body = truncate(draft.body, BODY_LIMIT) if draft.access == "free" else ""
            parts = [title, description, *draft.extra_text, body]
            index_rows.append(
                {
                    "item_id": item.item_id,
                    "item_type": item.item_type,
                    "access": item.access,
                    "index_text": "\n".join(_unique(part.strip() for part in parts)),
                }
            )
        return items, index_rows


def _format(item_type: str, duration: int | None) -> str:
    if item_type == "article":
        return (
            "quick_answer" if duration is not None and duration <= QUICK_ANSWER_MAX_MIN else "guide"
        )
    return {
        "glossary": "quick_answer",
        "document": "template",
        "calculator": "tool",
        "course": "course",
        "webinar": "recording",
        "podcast": "podcast",
    }[item_type]


def build_catalog(
    tables: Tables, base_url: str = DEFAULT_BASE_URL
) -> tuple[list[Item], list[dict[str, str]]]:
    return CatalogBuilder(tables, base_url).build()


def write_index_text(rows: list[dict[str, str]], path: Path) -> None:
    write_parquet(pa.Table.from_pylist(rows, schema=INDEX_TEXT_SCHEMA), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--index-text", type=Path, default=DEFAULT_INDEX_TEXT)
    parser.add_argument(
        "--base-url", default=os.environ.get("PLATFORM_BASE_URL") or DEFAULT_BASE_URL
    )
    args = parser.parse_args(argv)
    if not args.dump.is_file():
        print(f"Dump not found: {args.dump}", file=sys.stderr)
        return 1
    items, index_rows = build_catalog(read_tables(args.dump, TABLES), args.base_url)
    save_catalog(items, args.catalog)
    write_index_text(index_rows, args.index_text)
    counts = Counter(item.item_type for item in items)
    print(
        f"{len(items)} items: "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    )
    print(f"Catalog: {args.catalog}\nIndex text: {args.index_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
