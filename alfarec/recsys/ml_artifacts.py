"""Validated B04 artifacts shared by the CPU service and offline preparation."""

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from alfarec.recsys.retrieval import DenseIndex, RetrievalError, digest

DEFAULT_ML_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "ml"


def checked_manifest(path: Path, catalog_path: Path, ids: list[str]) -> dict:
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    if not isinstance(manifest, dict):
        raise RetrievalError("Invalid ML manifest")
    if (
        type(manifest.get("version")) is not int
        or manifest["version"] != 1
        or manifest.get("catalog_sha256") != digest(catalog_path)
    ):
        raise RetrievalError("ML catalog/version mismatch")
    if manifest.get("item_ids") != ids or len(set(ids)) != len(ids):
        raise RetrievalError("ML IDs mismatch")
    model_files = manifest.get("model_files")
    if (
        not isinstance(model_files, dict)
        or not model_files
        or not all(isinstance(name, str) for name in model_files)
        or "config.json" not in model_files
        or "modules.json" not in model_files
        or not any(name.endswith(".safetensors") for name in model_files)
        or not isinstance(manifest.get("revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
        or not isinstance(manifest.get("e5"), bool)
        or not isinstance(manifest.get("model_id"), str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", manifest["model_id"])
    ):
        raise RetrievalError("Invalid model manifest")
    for name, checksum in model_files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or relative.anchor
            or ":" in name
            or "\\" in name
            or ".." in relative.parts
            or relative.suffix not in {".json", ".safetensors", ".txt", ".model"}
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise RetrievalError("Invalid model file manifest")
    required = {
        "item_vectors.npy",
        "item_labels.parquet",
        "item_graph.npz",
        "semantic_graph.npz",
        "behaviour_graph.npz",
    }
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != required:
        raise RetrievalError("ML file manifest mismatch")
    for name, checksum in manifest["files"].items():
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise RetrievalError("Invalid artifact checksum")
        if digest(path / name) != checksum:
            raise RetrievalError(f"ML checksum mismatch: {name}")
    return manifest


def load_labels(path: Path, ids, taxonomy):
    rows = pq.read_table(path).to_pylist()
    if len(rows) != len(ids) or [row["item_id"] for row in rows] != list(ids):
        raise RetrievalError("ML label IDs mismatch")
    for row in rows:
        for field, vocab in (
            ("stages", taxonomy.stages),
            ("tasks", taxonomy.tasks),
            ("industries", taxonomy.industries),
        ):
            if not isinstance(row[field], list) or not set(row[field]) <= set(vocab):
                raise RetrievalError(f"Invalid ML {field}")
        if row["level"] is not None and row["level"] not in taxonomy.levels:
            raise RetrievalError("Invalid ML level")
    return {row["item_id"]: row for row in rows}


def load_graph(path: Path, count: int):
    graph = sp.load_npz(path).tocsr()
    if (
        graph.shape != (count, count)
        or not np.isfinite(graph.data).all()
        or (graph.data < 0).any()
        or (graph.data > 1).any()
    ):
        raise RetrievalError("Invalid ML graph")
    if graph.diagonal().any():
        raise RetrievalError("Graph contains self edges")
    return graph


def load_dense(path: Path, manifest: dict, model_path: Path):
    if not model_path.is_dir():
        return None
    actual = {
        f.relative_to(model_path).as_posix()
        for f in model_path.rglob("*")
        if f.is_file()
        and f.suffix in {".json", ".safetensors", ".txt", ".model"}
        and ".cache" not in f.relative_to(model_path).parts
    }
    if actual != set(manifest["model_files"]):
        raise RetrievalError("Model files differ from manifest")
    if any(f.suffix in {".bin", ".pkl", ".pickle", ".py", ".pt"} for f in model_path.rglob("*")):
        raise RetrievalError("Unsafe model files")
    for name, checksum in manifest["model_files"].items():
        relative = Path(name)
        file = model_path / relative
        if (
            relative.is_absolute()
            or relative.drive
            or ".." in relative.parts
            or not file.resolve().is_relative_to(model_path.resolve())
            or any(part.is_symlink() for part in [file, *file.parents] if part != model_path)
        ):
            raise RetrievalError("Invalid model file path")
        if digest(model_path / relative) != checksum:
            raise RetrievalError(f"Model checksum mismatch: {name}")
    from alfarec.recsys.encoder import SentenceEncoder

    encoder = SentenceEncoder(model_path, e5=manifest["e5"])
    vectors = np.load(path / "item_vectors.npy", allow_pickle=False)
    if vectors.dtype != np.float16:
        raise RetrievalError("Expected float16 item vectors")
    return DenseIndex(manifest["item_ids"], vectors, encoder)
