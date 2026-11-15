import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from alfarec.eval.label_check import load_manual, parquet_labels
from alfarec.pipeline.package_ml import graphs
from alfarec.pipeline.select_encoder import select
from alfarec.recsys.ml_artifacts import checked_manifest, load_graph, load_labels
from alfarec.recsys.retrieval import RetrievalError, digest
from alfarec.recsys.taxonomy import load_taxonomy


def test_graph_separates_semantic_and_behaviour_without_self_edges():
    vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]])
    rows = [{"user_key": "a", "course_id": "x"}, {"user_key": "a", "course_id": "z"}]
    semantic, behaviour = graphs(vectors, ["x", "y", "z"], rows)
    assert semantic[0, 1] == pytest.approx(0.8)
    assert behaviour[0, 2] == 1 and behaviour[0, 1] == 0
    assert not semantic.diagonal().any() and not behaviour.diagonal().any()


def test_bad_graph_is_rejected(tmp_path):
    path = tmp_path / "graph.npz"
    sp.save_npz(path, sp.eye(2, format="csr"))
    with pytest.raises(RetrievalError, match="self edges"):
        load_graph(path, 2)
    sp.save_npz(path, sp.csr_matrix([[0, np.nan], [0, 0]]))
    with pytest.raises(RetrievalError, match="Invalid"):
        load_graph(path, 2)


def test_duplicate_manual_and_predicted_labels_are_rejected(tmp_path):
    sample = {
        "items": [{"item_id": "x"}],
        "sample_sha256": "h",
        "levels": {},
        "tasks": {},
        "stages": {},
    }
    answer = {"item_id": "x", "level": None, "tasks": [], "stages": []}
    path = tmp_path / "manual.json"
    path.write_text(json.dumps({"sample_sha256": "h", "answers": [answer, answer]}))
    with pytest.raises(ValueError, match="Expected answers"):
        load_manual(path, sample)
    path = tmp_path / "labels.parquet"
    pq.write_table(pa.Table.from_pylist([answer, answer]), path)
    with pytest.raises(ValueError, match="Duplicate"):
        parquet_labels(path)


def test_labels_must_cover_exact_catalog_and_vocabulary(tmp_path):
    row = {"item_id": "x", "level": "expert", "stages": [], "tasks": [], "industries": []}
    path = tmp_path / "labels.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(RetrievalError, match="IDs"):
        load_labels(path, ["x", "y"], load_taxonomy())
    with pytest.raises(RetrievalError, match="level"):
        load_labels(path, ["x"], load_taxonomy())


def test_manifest_rejects_mismatched_ids_and_unlisted_artifacts(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.write_bytes(b"catalog")
    manifest = {"version": 1, "catalog_sha256": digest(catalog), "item_ids": ["x"], "files": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RetrievalError, match="IDs"):
        checked_manifest(tmp_path, catalog, ["y"])
    with pytest.raises(RetrievalError, match="model manifest"):
        checked_manifest(tmp_path, catalog, ["x"])


def test_float16_duplicate_vectors_roundtrip_graph(tmp_path):
    rng = np.random.default_rng(21)
    vector = rng.normal(size=384).astype(np.float16)
    semantic, _ = graphs(np.array([vector, vector, vector]), ["a", "b", "c"], [])
    path = tmp_path / "graph.npz"
    sp.save_npz(path, semantic)
    assert load_graph(path, 3).nnz == 6


def test_encoder_selection_ignores_holdout_and_excludes_slow_or_large_models():
    def candidate(name, tune, held, *, size=100, latency=10):
        return {
            "candidate": name,
            "weights_bytes": size,
            "cpu_p95_ms": latency,
            "metrics": {"tune_ndcg": {"mean": tune}, "holdout_ndcg": {"mean": held}},
        }

    eligible = candidate("small", 0.4, 0.1)
    manifests = [
        eligible,
        candidate("tiny", 0.3, 1.0),
        candidate("large", 0.9, 1.0, size=600_000_000),
        candidate("slow", 0.9, 1.0, latency=200),
        candidate("tfidf", 1, 1),
    ]
    assert select(manifests) is eligible
    with pytest.raises(ValueError, match="No neural"):
        select([candidate("slow", 1, 1, latency=200)])


@pytest.mark.parametrize(
    "unsafe", ["../weights.safetensors", "C:weights.safetensors", "/weights.safetensors"]
)
def test_model_manifest_rejects_unsafe_paths(tmp_path, unsafe):
    catalog = tmp_path / "catalog"
    catalog.write_bytes(b"catalog")
    manifest = {
        "version": 1,
        "catalog_sha256": digest(catalog),
        "item_ids": ["x"],
        "model_id": "test/model",
        "revision": "a" * 40,
        "e5": False,
        "model_files": {"config.json": "b" * 64, "modules.json": "b" * 64, unsafe: "b" * 64},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RetrievalError, match="model file"):
        checked_manifest(tmp_path, catalog, ["x"])
