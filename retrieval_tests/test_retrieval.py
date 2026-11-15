import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from alfarec.eval.benchmark_retrieval import benchmark, render_report
from alfarec.pipeline.build_index import build_index, main, prepare_texts
from alfarec.recsys.catalog import load_catalog, save_catalog
from alfarec.recsys.contracts import Item
from alfarec.recsys.retrieval import (
    DenseIndex,
    RetrievalError,
    TfidfIndex,
    choose_index,
    digest,
    texts_digest,
    write_json,
)


@pytest.fixture
def items() -> list[Item]:
    return [
        Item(
            item_id="tax",
            title="Налоги и отчётность",
            summary="Расчёт налогов для бизнеса",
            item_type="article",
            format="guide",
            access="free",
            url="https://example.test/tax",
        ),
        Item(
            item_id="marketing",
            title="Продвижение бизнеса",
            summary="Реклама и привлечение клиентов",
            item_type="article",
            format="guide",
            access="free",
            url="https://example.test/marketing",
        ),
        Item(
            item_id="team",
            title="Найм команды",
            summary="Поиск и оформление сотрудников",
            item_type="article",
            format="guide",
            access="free",
            url="https://example.test/team",
        ),
    ]


@pytest.fixture
def catalog(tmp_path: Path, items: list[Item]) -> Path:
    path = tmp_path / "catalog.parquet"
    save_catalog(items, path)
    return path


def test_metadata_build_load_and_retrieval_match(catalog: Path, tmp_path: Path) -> None:
    index = build_index(catalog, tmp_path / "index", source="metadata")
    loaded = TfidfIndex.load(
        tmp_path / "index",
        expected_item_ids=index.item_ids,
        expected_catalog_sha256=digest(catalog),
    )
    expected = index.search("налоговая отчётность")
    assert expected[0].item_id == "tax"
    assert loaded.search("налоговая отчётность") == expected
    assert loaded.search("РЕКЛАМА для клиентов")[0].item_id == "marketing"
    assert loaded.search("оформление сотрудников")[0].item_id == "team"
    assert loaded.mode == "tfidf"


def test_empty_unknown_and_excluded_queries(catalog: Path, tmp_path: Path) -> None:
    index = build_index(catalog, tmp_path / "index", source="metadata")
    assert index.search("   ") == []
    assert index.search("㑻㑼㑽㑾") == []
    assert all(hit.item_id != "tax" for hit in index.search("налоги", excluded={"tax"}))
    assert index.search("налоги", excluded=set(index.item_ids)) == []


def test_ties_have_stable_id_order() -> None:
    index = TfidfIndex.fit(["b", "a"], ["одинаковый текст", "одинаковый текст"])
    assert [hit.item_id for hit in index.search("одинаковый текст")] == ["a", "b"]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_is_rejected_even_for_empty_query(limit: object) -> None:
    index = TfidfIndex.fit(["a"], ["налоги"])
    with pytest.raises(ValueError, match="Limit"):
        index.search("", limit=limit)


def test_build_is_reproducible(catalog: Path, tmp_path: Path) -> None:
    for name in ("first", "second"):
        build_index(catalog, tmp_path / name, source="metadata")
    first, second = tmp_path / "first", tmp_path / "second"
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }


def test_stale_catalog_and_id_order_are_rejected(catalog: Path, tmp_path: Path) -> None:
    index = build_index(catalog, tmp_path / "index", source="metadata")
    with pytest.raises(RetrievalError, match="different catalog"):
        TfidfIndex.load(tmp_path / "index", expected_catalog_sha256="0" * 64)
    with pytest.raises(RetrievalError, match="IDs or order"):
        TfidfIndex.load(tmp_path / "index", expected_item_ids=list(reversed(index.item_ids)))


@pytest.mark.parametrize("filename", ["matrix.npz", "idf.npy", "vocabulary.json"])
def test_corruption_is_detected_before_deserialization(
    catalog: Path, tmp_path: Path, filename: str
) -> None:
    build_index(catalog, tmp_path / "index", source="metadata")
    (tmp_path / "index" / filename).write_bytes(b"corrupted")
    with pytest.raises(RetrievalError, match="checksum mismatch"):
        TfidfIndex.load(tmp_path / "index")


def test_dependency_version_mismatch_is_rejected(catalog: Path, tmp_path: Path) -> None:
    build_index(catalog, tmp_path / "index", source="metadata")
    manifest_path = tmp_path / "index" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["versions"]["sklearn"] = "old-version"
    write_json(manifest_path, manifest)
    with pytest.raises(RetrievalError, match="versions differ"):
        TfidfIndex.load(tmp_path / "index")


def test_default_source_does_not_silently_use_metadata(
    catalog: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "output"
    assert main(["--catalog", str(catalog), "--output", str(output)]) == 1
    assert not output.exists()
    assert "requires an explicit text file" in capsys.readouterr().err


def test_existing_output_is_preserved(catalog: Path, tmp_path: Path) -> None:
    output = tmp_path / "index"
    build_index(catalog, output, source="metadata")
    checksum = digest(output / "manifest.json")
    with pytest.raises(FileExistsError):
        build_index(catalog, output, source="metadata")
    assert digest(output / "manifest.json") == checksum


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "access", "type", "empty"])
def test_text_source_must_match_catalog(catalog: Path, tmp_path: Path, mutation: str) -> None:
    items = load_catalog(catalog)
    rows = [
        dict(item_id=i.item_id, item_type=i.item_type, access=i.access, index_text=i.title)
        for i in items
    ]
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({**rows[0], "item_id": "extra"})
    elif mutation == "access":
        rows[0]["access"] = "paid"
    elif mutation == "type":
        rows[0]["item_type"] = "course"
    else:
        rows[0]["index_text"] = ""
    text_path = tmp_path / "texts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), text_path)
    with pytest.raises(ValueError):
        prepare_texts(items, source="index_text", index_text=text_path)


def test_full_text_source_respects_item_order_and_provenance(catalog: Path, tmp_path: Path) -> None:
    items = load_catalog(catalog)
    rows = [
        dict(item_id=i.item_id, item_type=i.item_type, access=i.access, index_text=i.title)
        for i in items
    ]
    text_path = tmp_path / "texts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), text_path)
    index = build_index(catalog, tmp_path / "index", source="index_text", index_text=text_path)
    assert index.item_ids == tuple(sorted(i.item_id for i in items))
    manifest = json.loads((tmp_path / "index" / "manifest.json").read_text("utf-8"))
    assert manifest["text_source"] == "index_text"


class FixedEncoder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.vector for _ in texts], dtype=np.float32)


def test_dense_encoder_interface_cosine_and_fallback() -> None:
    dense = DenseIndex(["b", "a"], np.array([[1, 0], [0, 1]]), FixedEncoder([0, 2]))
    assert dense.search("query")[0].item_id == "a"
    assert dense.search("query")[0].score == pytest.approx(1)
    assert dense.search("query", excluded={"a"}) == []
    fallback = TfidfIndex.fit(["b", "a"], ["налоги", "реклама"])
    assert choose_index(dense, fallback).mode == "dense"
    assert choose_index(None, fallback).mode == "tfidf"
    with pytest.raises(RetrievalError, match="different items"):
        choose_index(dense, TfidfIndex.fit(["other"], ["налоги"]))


@pytest.mark.parametrize("vector", [[float("nan"), 1], [1], [float("inf"), 0]])
def test_invalid_encoder_output_is_rejected(vector: list[float]) -> None:
    dense = DenseIndex(["a"], np.array([[1, 0]]), FixedEncoder(vector))
    with pytest.raises(RetrievalError):
        dense.search("query")


def test_dense_index_does_not_mutate_input_and_handles_large_finite_weights() -> None:
    vectors = np.array([[1e38, 0]], dtype=np.float32)
    original = vectors.copy()
    dense = DenseIndex(["a"], vectors, FixedEncoder([1e38, 0]))
    np.testing.assert_array_equal(vectors, original)
    assert dense.search("query")[0].score == pytest.approx(1)


@pytest.mark.parametrize("ids", [[" a "], ["a", " a "], [""], ["x" * 129]])
def test_item_ids_are_canonical_before_persistence(ids: list[str]) -> None:
    with pytest.raises(RetrievalError, match="item ID"):
        TfidfIndex.fit(ids, ["налоги"] * len(ids))


def test_external_vectorizer_cannot_change_a_built_index() -> None:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        strip_accents="unicode",
        max_features=50_000,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(["налоги отчётность", "реклама клиенты"]).tocsr()
    index = TfidfIndex(["tax", "marketing"], vectorizer, matrix)
    expected = index.search("налоги")
    vectorizer.fit(["другие признаки"])
    assert index.search("налоги") == expected


@pytest.mark.parametrize("parameters", [{}, {"max_features": 100}, {"norm": None}])
def test_external_unsupported_vectorizer_cannot_create_an_unloadable_index(
    parameters: dict,
) -> None:
    vectorizer = TfidfVectorizer(**parameters)
    matrix = vectorizer.fit_transform(["financial services", "market research"]).tocsr()
    with pytest.raises(RetrievalError, match="Unsupported vectorizer"):
        TfidfIndex(["finance", "marketing"], vectorizer, matrix)


def test_loaded_index_exposes_and_checks_text_provenance(catalog: Path, tmp_path: Path) -> None:
    output = tmp_path / "index"
    build_index(catalog, output, source="metadata")
    loaded = TfidfIndex.load(output, expected_text_source="metadata")
    assert loaded.provenance.text_source == "metadata"
    loaded.provenance.item_ids.append("changed externally")
    assert "changed externally" not in loaded.provenance.item_ids
    with pytest.raises(RetrievalError, match="text source differs"):
        TfidfIndex.load(output, expected_text_source="index_text")
    with pytest.raises(RetrievalError, match="different texts"):
        TfidfIndex.load(output, expected_texts_sha256="0" * 64)


def test_json_has_platform_independent_utf8_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_json(path, {"ключ": "значение"})
    assert path.read_bytes() == '{"ключ":"значение"}\n'.encode()


def test_committed_fallback_matches_committed_catalog() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = root / "artifacts" / "catalog.parquet"
    ids, texts = prepare_texts(load_catalog(catalog), source="metadata")
    index = TfidfIndex.load(
        root / "artifacts" / "retrieval_tfidf",
        expected_item_ids=ids,
        expected_catalog_sha256=digest(catalog),
        expected_text_source="metadata",
        expected_texts_sha256=texts_digest(ids, texts),
    )
    assert index.search(texts[0], limit=1)


@pytest.mark.parametrize("source", ["metadata", "index_text"])
def test_benchmark_report_describes_the_actual_source(
    catalog: Path, tmp_path: Path, source: str
) -> None:
    rows = [
        dict(item_id=i.item_id, item_type=i.item_type, access=i.access, index_text=i.title)
        for i in load_catalog(catalog)
    ]
    text_file = tmp_path / "texts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), text_file)
    output = tmp_path / "index"
    build_index(catalog, output, source=source, index_text=text_file)
    result = benchmark(output, catalog, source=source, index_text=text_file, repeats=1)
    report = render_report(result)
    assert f"источник `{source}`" in report
    assert f"--source {source}" in report
    if source == "index_text":
        assert "публичных метаданных" not in report
        assert "--index-text" in report
    assert result["encode_p95_ms"] >= 0
    assert result["search_p95_ms"] >= 0
