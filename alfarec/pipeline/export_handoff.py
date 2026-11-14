"""Export the B02 inputs for B04: texts, editorial links and course enrolments.

Usage: python -m alfarec.pipeline.export_handoff [--dump PATH] [--output DIR]

Files go to data/handoff/ (outside git) together with manifest.json. The committed report
reports/b02_handoff.md keeps only counts and hashes, so the receiver can check the files.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa

from alfarec.pipeline.build_catalog import (
    DEFAULT_DUMP,
    TABLES,
    _parse_timestamp,
    build_catalog,
    write_index_text,
)
from alfarec.pipeline.dump_reader import Tables, read_tables
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog, write_parquet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "handoff"
DEFAULT_REPORT = ROOT / "reports" / "b02_handoff.md"
VERSION = 1
# Platform user ids are replaced by stable pseudonyms; the original ids never leave the dump.
USER_SALT = "alfarec-handoff-v1:"
SOURCE_TABLES = ("Articles", "Documents", "Courses", "Webinars", "Podcasts", "Glossary")
EXTRA_TABLES = (
    "RecommendedItems", "DocumentBindings", "CourseUserBindings", "Landings", "Lessons",
    "Calculators",
)  # fmt: skip

EDITORIAL_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("order", pa.int32()),
    ]
)
ENROLMENT_SCHEMA = pa.schema(
    [
        pa.field("user_key", pa.string(), nullable=False),
        pa.field("course_id", pa.string(), nullable=False),
        pa.field("status", pa.int8(), nullable=False),
        pa.field("enrolled_at", pa.timestamp("us", tz="UTC")),
        pa.field("completed_at", pa.timestamp("us", tz="UTC")),
        pa.field("is_paid", pa.bool_(), nullable=False),
    ]
)

# Where each part of index_text comes from and why it may be used (spec 5.1, card B02).
TEXT_SOURCES = [
    ("Заголовок и описание всех материалов", "Публичная карточка материала в каталоге платформы"),
    ("`ContentHeadings` статей", "Оглавление открытой статьи"),
    (
        "Тело открытых статей, документов, глоссария и калькуляторов, до 4000 символов",
        "Страница открыта без подписки; у статей с `IsPublic = false` тело не берётся",
    ),
    (
        "`Target`, `DescriptionAdditional`, `CourseInfoBindings`, лендинг курса",
        "Лендинг курса: «Кому подойдёт», «Что вы получите»; без уроков",
    ),
    ("`DescriptionExt` вебинаров и подкастов, `PodcastDetails`", "Описание на странице записи"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_key(user_id: str) -> str:
    return hashlib.sha256((USER_SALT + user_id).encode()).hexdigest()[:16]


def editorial_pairs(tables: Tables, catalog_ids: set[str]) -> list[dict]:
    """Editor-made "see also" links between catalog items; landings map to their course."""
    course_by_slug = {row["Url"]: row["Id"] for row in tables["Courses"]}
    landing_course = {
        row["Id"]: course_by_slug[row["Url"]]
        for row in tables["Landings"]
        if row["Url"] in course_by_slug
    }
    rows = []
    for row in tables["RecommendedItems"]:
        source = landing_course.get(row["Id"], row["Id"])
        if source in catalog_ids and row["ItemId"] in catalog_ids and source != row["ItemId"]:
            order = int(row["Order"]) if row["Order"] is not None else None
            rows.append(
                {"source": "recommended_items", "source_id": source, "item_id": row["ItemId"],
                 "order": order}
            )  # fmt: skip
    for row in tables["DocumentBindings"]:
        if row["DocumentId"] in catalog_ids and row["LinkedDocumentId"] in catalog_ids:
            order = int(row["Order"]) if row["Order"] is not None else None
            rows.append(
                {"source": "document_bindings", "source_id": row["DocumentId"],
                 "item_id": row["LinkedDocumentId"], "order": order}
            )  # fmt: skip
    unique = {(row["source"], row["source_id"], row["item_id"]): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def enrolments(tables: Tables, catalog_ids: set[str]) -> list[dict]:
    rows = []
    for row in tables["CourseUserBindings"]:
        if row["IsDeleted"] == "t" or row["CourseId"] not in catalog_ids:
            continue
        rows.append(
            {
                "user_key": user_key(row["UserId"]),
                "course_id": row["CourseId"],
                "status": int(row["Status"]),
                "enrolled_at": _parse_timestamp(row["Created"]),
                "completed_at": _parse_timestamp(row["Completed"]),
                "is_paid": row["IsPaid"] == "t",
            }
        )
    rows.sort(key=lambda row: (row["user_key"], row["course_id"], str(row["enrolled_at"])))
    return rows


def export(dump: Path, output: Path, catalog_path: Path) -> dict:
    tables = read_tables(dump, sorted(set(TABLES) | set(SOURCE_TABLES) | set(EXTRA_TABLES)))
    items, index_rows = build_catalog(tables)
    committed = {item.item_id for item in load_catalog(catalog_path)}
    if {item.item_id for item in items} != committed:
        raise ValueError("The dump does not produce the committed catalog; rebuild it first")
    ids = committed
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "index_text.parquet": len(index_rows),
        "editorial_pairs.parquet": 0,
        "course_enrolments.parquet": 0,
    }
    write_index_text(index_rows, output / "index_text.parquet")
    pairs = editorial_pairs(tables, ids)
    write_parquet(
        pa.Table.from_pylist(pairs, schema=EDITORIAL_SCHEMA), output / "editorial_pairs.parquet"
    )
    files["editorial_pairs.parquet"] = len(pairs)
    enrolled = enrolments(tables, ids)
    write_parquet(
        pa.Table.from_pylist(enrolled, schema=ENROLMENT_SCHEMA),
        output / "course_enrolments.parquet",
    )
    files["course_enrolments.parquet"] = len(enrolled)
    manifest = {
        "version": VERSION,
        "catalog_sha256": _sha256(catalog_path),
        "dump_sha256": _sha256(dump),
        "files": {
            name: {"rows": rows, "sha256": _sha256(output / name)} for name, rows in files.items()
        },
        "editorial_sources": dict(Counter(row["source"] for row in pairs)),
        "enrolment_users": len({row["user_key"] for row in enrolled}),
        "enrolment_status": {
            str(k): v for k, v in sorted(Counter(r["status"] for r in enrolled).items())
        },
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(text, encoding="utf-8", newline="\n")
    return manifest


def render_report(manifest: dict) -> str:
    lines = [
        "# Передача входов B02 → B04",
        "",
        "Отчёт создаётся командой `python -m alfarec.pipeline.export_handoff` вместе с файлами в"
        " `data/handoff/`. Сами файлы в git не попадают и передаются вне репозитория; по этому"
        " отчёту получатель сверяет версии и хеши.",
        "",
        f"- Каталог: `artifacts/catalog.parquet`, SHA-256 `{manifest['catalog_sha256']}`.",
        f"- Дамп: SHA-256 `{manifest['dump_sha256']}`.",
        f"- Версия формата передачи: {manifest['version']}.",
        "",
        "| Файл | Строк | SHA-256 | Содержимое |",
        "|---|---|---|---|",
    ]
    descriptions = {
        "index_text.parquet": "`item_id`, `item_type`, `access`, `index_text` — текст"
        " для эмбеддингов",
        "editorial_pairs.parquet": "`source`, `source_id`, `item_id`, `order` — редакторские связи",
        "course_enrolments.parquet": "`user_key`, `course_id`, `status`, `enrolled_at`,"
        " `completed_at`, `is_paid` — записи на курсы",
    }
    for name, entry in manifest["files"].items():
        lines.append(f"| `{name}` | {entry['rows']} | `{entry['sha256']}` | {descriptions[name]} |")
    sources = ", ".join(f"{k} {v}" for k, v in sorted(manifest["editorial_sources"].items()))
    statuses = ", ".join(f"{k} — {v}" for k, v in manifest["enrolment_status"].items())
    lines += [
        "",
        "## Правила",
        "",
        f"- Редакторские связи ({sources}): обе стороны есть в каталоге, связь с самим собой"
        " исключена. `RecommendedItems.Id` — ID исходного материала; подборки лендингов"
        " сопоставлены с курсом по слагу. Связи для проверки моделей, не признак ранжирования.",
        f"- Записи на курсы: {manifest['enrolment_users']} пользователей, статусы {statuses}"
        " (0 — не начат, 1 — в процессе, 2 — завершён). Неудалённые записи на курсы каталога."
        " ID платформы заменены на `user_key` — первые 16 знаков SHA-256 с фиксированной солью;"
        " даты в данных точны до месяца.",
        "- ID материалов во всех файлах совпадают с `item_id` каталога этой версии.",
        "",
        "## Публичность полей текста индекса",
        "",
        "| Источник | Основание |",
        "|---|---|",
    ]
    lines += [f"| {source} | {reason} |" for source, reason in TEXT_SOURCES]
    lines += [
        "",
        "Тексты уроков не читаются, тело статей «Курс+» не берётся; тест на реальном дампе"
        " проверяет отсутствие строк уроков в артефактах и ответах API. Публичность страниц"
        " определена по данным дампа (статусы, `IsClosed`, `IsPublic`, тип курса) и на сайте"
        " платформы не проверялась.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if not args.dump.is_file():
        print(f"Dump not found: {args.dump}", file=sys.stderr)
        return 1
    manifest = export(args.dump, args.output, args.catalog)
    args.report.write_text(render_report(manifest), encoding="utf-8", newline="\n")
    print(f"Handoff: {args.output}; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
