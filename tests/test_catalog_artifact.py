"""Checks of the committed catalog artifact and, when the dump is present, of its rebuild."""

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from alfarec.pipeline.build_catalog import DEFAULT_DUMP, main
from alfarec.pipeline.coverage import DEFAULT_REPORT, compute_coverage, render_report
from alfarec.pipeline.dump_reader import read_tables
from alfarec.pipeline.text import html_to_text
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.taxonomy import load_taxonomy

# Published, non-deleted materials per type, as summarised in docs/hackathon/DATA.md.
EXPECTED_COUNTS = {
    "article": 838,
    "document": 247,
    "course": 33,
    "webinar": 13,
    "podcast": 8,
    "glossary": 351,
    "calculator": 7,
}


def test_committed_catalog_matches_the_data_summary():
    items = load_catalog(DEFAULT_CATALOG_PATH)
    assert Counter(item.item_type for item in items) == EXPECTED_COUNTS
    assert len({item.item_id for item in items}) == len(items)
    assert all(item.title.strip() and str(item.url).startswith("https://") for item in items)
    assert {item.access for item in items if item.item_type in {"glossary", "document"}} == {"free"}
    paid = [item for item in items if item.access == "paid"]
    assert [(item.item_type, item.price_rub) for item in paid] == [("course", 3990)]


def test_taxonomy_links_resolve_and_coverage_report_is_current():
    taxonomy = load_taxonomy()
    coverage = compute_coverage(taxonomy, load_catalog(DEFAULT_CATALOG_PATH))
    assert taxonomy.source_links_status == "audited"
    assert coverage.missing == []
    assert DEFAULT_REPORT.read_text(encoding="utf-8") == render_report(taxonomy, coverage)


@pytest.mark.skipif(not DEFAULT_DUMP.is_file(), reason="the dump is handed over outside git")
def test_rebuild_from_dump_is_reproducible_and_has_no_lesson_text(tmp_path: Path):
    catalog, index = tmp_path / "catalog.parquet", tmp_path / "index_text.parquet"
    base_url = "https://kurs.alfabank.ru"
    arguments = ["--catalog", str(catalog), "--index-text", str(index), "--base-url", base_url]
    assert main(arguments) == 0
    assert catalog.read_bytes() == DEFAULT_CATALOG_PATH.read_bytes()

    tables = read_tables(
        DEFAULT_DUMP, ["LessonEmbedItems", "Articles", "ArticleEmbedItems", "Statuses"]
    )
    lesson_lines = {
        line
        for entry in tables["LessonEmbedItems"]
        for line in html_to_text(entry["HtmlContent"]).splitlines()
        if len(line) >= 40
    }
    # Some lessons quote open articles word for word (tax brackets, legal lists). Such a line
    # is public anyway, so only lesson text absent from open articles counts as a leak.
    published = next(row["Id"] for row in tables["Statuses"] if row["Title"] == "Опубликовано")
    open_articles = {
        row["Id"]
        for row in tables["Articles"]
        if row["IsClosed"] != "t" and row["IsDeleted"] != "t" and row["StatusId"] == published
    }
    public = "\n".join(
        html_to_text(entry["HtmlContent"])
        for entry in tables["ArticleEmbedItems"]
        if entry["ArticleId"] in open_articles
    )
    texts = [entry["index_text"] for entry in pq.read_table(index).to_pylist()]
    texts += [item.model_dump_json() for item in load_catalog(catalog)]
    corpus = "\n".join(texts)
    leaked = [line for line in lesson_lines if line in corpus and line not in public]
    assert not leaked
