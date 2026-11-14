from pathlib import Path

import pyarrow.parquet as pq
import pytest

from alfarec.pipeline.build_catalog import DEFAULT_DUMP
from alfarec.pipeline.export_handoff import (
    DEFAULT_REPORT,
    editorial_pairs,
    enrolments,
    main,
)


def tables() -> dict:
    return {
        "Courses": [{"Id": "course-1", "Url": "course-slug"}],
        "Landings": [{"Id": "landing-1", "Url": "course-slug"}],
        "RecommendedItems": [
            {"Id": "article-1", "ItemId": "article-2", "Order": "1"},
            {"Id": "article-1", "ItemId": "article-1", "Order": "2"},
            {"Id": "landing-1", "ItemId": "article-2", "Order": None},
            {"Id": "article-1", "ItemId": "draft-9", "Order": "0"},
        ],
        "DocumentBindings": [
            {"DocumentId": "doc-1", "LinkedDocumentId": "doc-2", "Order": "0"},
            {"DocumentId": "doc-1", "LinkedDocumentId": "gone", "Order": "1"},
        ],
        "CourseUserBindings": [
            {"UserId": "platform-user-1", "CourseId": "course-1", "Status": "2", "IsDeleted": "f",
             "Created": "2026-03-01 00:00:00+00", "Completed": "2026-04-01 00:00:00+00",
             "IsPaid": "f"},
            {"UserId": "platform-user-2", "CourseId": "course-1", "Status": "0", "IsDeleted": "t",
             "Created": "2026-03-01 00:00:00+00", "Completed": None, "IsPaid": "f"},
        ],
    }  # fmt: skip


CATALOG = {"article-1", "article-2", "course-1", "doc-1", "doc-2"}


def test_editorial_pairs_keep_catalog_links_and_map_landings():
    pairs = editorial_pairs(tables(), CATALOG)
    assert [(p["source"], p["source_id"], p["item_id"]) for p in pairs] == [
        ("document_bindings", "doc-1", "doc-2"),
        ("recommended_items", "article-1", "article-2"),
        ("recommended_items", "course-1", "article-2"),
    ]


def test_enrolments_hide_platform_ids():
    rows = enrolments(tables(), CATALOG)
    assert len(rows) == 1
    assert rows[0]["user_key"] == "u00001"
    assert (rows[0]["status"], rows[0]["is_paid"]) == (2, False)


@pytest.mark.skipif(not DEFAULT_DUMP.is_file(), reason="the dump is handed over outside git")
def test_export_is_reproducible_and_matches_committed_report(tmp_path: Path):
    reports = []
    for run in range(2):
        report = tmp_path / f"report-{run}.md"
        assert main(["--output", str(tmp_path / f"out-{run}"), "--report", str(report)]) == 0
        reports.append(report.read_text(encoding="utf-8"))
    assert reports[0] == reports[1] == DEFAULT_REPORT.read_text(encoding="utf-8")
    enrolled = pq.read_table(tmp_path / "out-0" / "course_enrolments.parquet").to_pylist()
    raw = (tmp_path / "out-0" / "course_enrolments.parquet").read_bytes()
    assert enrolled and b"platform" not in raw
