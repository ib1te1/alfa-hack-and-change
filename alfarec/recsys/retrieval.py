import copy
import hashlib
import json
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from zipfile import BadZipFile

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from pydantic import Field, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer

from alfarec.recsys.contracts import Contract, Identifier

FILES = ("matrix.npz", "idf.npy", "vocabulary.json")


class RetrievalError(ValueError):
    """An index artifact is invalid or incompatible with its inputs."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_bytes(content.encode("utf-8"))


def texts_digest(item_ids: Sequence[str], texts: Sequence[str]) -> str:
    content = json.dumps(
        list(zip(item_ids, texts, strict=True)), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def versions() -> dict[str, str]:
    return {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


class IndexManifest(Contract):
    version: Literal[1]
    mode: Literal["tfidf"]
    text_source: Literal["metadata", "index_text"]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    texts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_ids: list[Identifier] = Field(min_length=1)
    features: int = Field(ge=1, strict=True)
    max_features: int = Field(ge=1, strict=True)
    versions: dict[str, str]
    files: dict[str, str]

    @field_validator("item_ids", mode="before")
    @classmethod
    def validate_canonical_ids(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("Manifest item_ids must be a list")
        _ids(value)
        return value


@dataclass(frozen=True)
class SearchHit:
    item_id: str
    score: float


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _ids(item_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(item_ids)
    if any(
        not isinstance(item_id, str) or not 1 <= len(item_id) <= 128 or item_id != item_id.strip()
        for item_id in result
    ):
        raise RetrievalError("Invalid index item ID")
    if not result or len(set(result)) != len(result):
        raise RetrievalError("Index item IDs must be nonempty and unique")
    return result


def _hits(
    item_ids: tuple[str, ...], scores: np.ndarray, limit: int, excluded: Set[str]
) -> list[SearchHit]:
    if type(limit) is not int or limit < 1:
        raise ValueError("Limit must be a positive integer")
    if scores.shape != (len(item_ids),) or not np.isfinite(scores).all():
        raise RetrievalError("Invalid retrieval scores")
    entries = [
        SearchHit(item_id, float(score))
        for item_id, score in zip(item_ids, scores, strict=True)
        if item_id not in excluded and score > 0
    ]
    return sorted(entries, key=lambda entry: (-entry.score, entry.item_id))[:limit]


def _query(text: str, limit: int) -> str:
    if type(limit) is not int or limit < 1:
        raise ValueError("Limit must be a positive integer")
    if not isinstance(text, str):
        raise ValueError("Query must be a string")
    return text.strip()


def _vectorizer(max_features: int, vocabulary: dict[str, int] | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        vocabulary=vocabulary,
        dtype=np.float32,
        norm="l2",
    )


class TfidfIndex:
    mode = "tfidf"

    def __init__(
        self,
        item_ids: Sequence[str],
        vectorizer: TfidfVectorizer,
        matrix: sp.csr_matrix,
        manifest: IndexManifest | None = None,
    ) -> None:
        self.item_ids = _ids(item_ids)
        if type(vectorizer.max_features) is not int or vectorizer.max_features < 1:
            raise RetrievalError("Unsupported vectorizer configuration")
        parameters = vectorizer.get_params(deep=False)
        parameters["vocabulary"] = None
        if parameters != _vectorizer(vectorizer.max_features).get_params(deep=False):
            raise RetrievalError("Unsupported vectorizer configuration")
        if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
            raise RetrievalError("Vectorizer must be fitted")
        self._vectorizer = copy.deepcopy(vectorizer)
        self._manifest = manifest.model_copy(deep=True) if manifest is not None else None
        self._matrix = matrix.copy().tocsr()
        self._matrix.check_format(full_check=True)
        if self._matrix.shape != (len(self.item_ids), len(vectorizer.vocabulary_)):
            raise RetrievalError("Matrix dimensions do not match index metadata")
        if not np.isfinite(self._matrix.data).all() or (self._matrix.data < 0).any():
            raise RetrievalError("Invalid TF-IDF weights")
        norms = np.sqrt(self._matrix.multiply(self._matrix).sum(axis=1)).A1
        if not np.allclose(norms, 1, atol=1e-5):
            raise RetrievalError("Document vectors must have unit norm")

    @property
    def provenance(self) -> IndexManifest | None:
        return self._manifest.model_copy(deep=True) if self._manifest is not None else None

    @classmethod
    def fit(
        cls, item_ids: Sequence[str], texts: Sequence[str], max_features: int = 50_000
    ) -> "TfidfIndex":
        _ids(item_ids)
        if len(item_ids) != len(texts) or any(not text.strip() for text in texts):
            raise RetrievalError("Every item must have one nonempty text")
        if type(max_features) is not int or max_features < 1:
            raise ValueError("max_features must be a positive integer")
        vectorizer = _vectorizer(max_features)
        return cls(item_ids, vectorizer, vectorizer.fit_transform(texts).tocsr())

    def encode_query(self, text: str) -> sp.csr_matrix:
        return self._vectorizer.transform([text]).tocsr()

    def search(
        self, text: str, limit: int = 10, excluded: Set[str] = frozenset()
    ) -> list[SearchHit]:
        text = _query(text, limit)
        if not text:
            return []
        query = self.encode_query(text)
        scores = (self._matrix @ query.T).toarray().ravel()
        return _hits(self.item_ids, scores, limit, excluded)

    def save(self, path: Path, *, text_source: str, catalog_sha256: str, texts_sha256: str) -> None:
        path.mkdir(parents=True, exist_ok=False)
        sp.save_npz(path / "matrix.npz", self._matrix, compressed=True)
        np.save(path / "idf.npy", self._vectorizer.idf_, allow_pickle=False)
        write_json(
            path / "vocabulary.json",
            {token: int(i) for token, i in self._vectorizer.vocabulary_.items()},
        )
        manifest = IndexManifest(
            version=1,
            mode="tfidf",
            text_source=text_source,
            catalog_sha256=catalog_sha256,
            texts_sha256=texts_sha256,
            item_ids=list(self.item_ids),
            features=self._matrix.shape[1],
            max_features=self._vectorizer.max_features,
            versions=versions(),
            files={name: digest(path / name) for name in FILES},
        )
        # The manifest is the completion marker and must be written last.
        write_json(path / "manifest.json", manifest.model_dump())

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_item_ids: Sequence[str] | None = None,
        expected_catalog_sha256: str | None = None,
        expected_text_source: Literal["metadata", "index_text"] | None = None,
        expected_texts_sha256: str | None = None,
    ) -> "TfidfIndex":
        try:
            manifest = IndexManifest.model_validate_json(
                (path / "manifest.json").read_text("utf-8")
            )
            if manifest.versions != versions():
                raise RetrievalError("Index dependency versions differ; rebuild the index")
            if set(manifest.files) != set(FILES):
                raise RetrievalError("Invalid index file list")
            for name in FILES:
                if digest(path / name) != manifest.files[name]:
                    raise RetrievalError(f"Index checksum mismatch: {name}")
            if expected_item_ids is not None and tuple(expected_item_ids) != tuple(
                manifest.item_ids
            ):
                raise RetrievalError("Index item IDs or order differ from the catalog")
            if (
                expected_catalog_sha256 is not None
                and expected_catalog_sha256 != manifest.catalog_sha256
            ):
                raise RetrievalError("Index was built for a different catalog")
            if expected_text_source is not None and expected_text_source != manifest.text_source:
                raise RetrievalError("Index text source differs from the required source")
            if expected_texts_sha256 is not None and expected_texts_sha256 != manifest.texts_sha256:
                raise RetrievalError("Index was built from different texts")
            vocabulary = json.loads((path / "vocabulary.json").read_text("utf-8"))
            if not isinstance(vocabulary, dict) or len(vocabulary) != manifest.features:
                raise RetrievalError("Invalid index vocabulary")
            if any(
                not isinstance(token, str) or type(i) is not int for token, i in vocabulary.items()
            ):
                raise RetrievalError("Invalid vocabulary entries")
            if sorted(vocabulary.values()) != list(range(manifest.features)):
                raise RetrievalError("Vocabulary indices must be contiguous")
            idf = np.load(path / "idf.npy", allow_pickle=False)
            if idf.shape != (manifest.features,) or not np.isfinite(idf).all() or (idf < 1).any():
                raise RetrievalError("Invalid inverse document frequencies")
            vectorizer = _vectorizer(manifest.max_features, vocabulary)
            vectorizer.fit([""])
            vectorizer.idf_ = idf
            return cls(manifest.item_ids, vectorizer, sp.load_npz(path / "matrix.npz"), manifest)
        except (OSError, ValueError, TypeError, KeyError, BadZipFile) as error:
            raise RetrievalError(f"Cannot load retrieval index at {path}: {error}") from error


class DenseIndex:
    """Cosine retrieval for an explicitly supplied, local vector encoder."""

    mode = "dense"

    def __init__(self, item_ids: Sequence[str], vectors: np.ndarray, encoder: TextEncoder) -> None:
        self.item_ids = _ids(item_ids)
        self._vectors = np.array(vectors, dtype=np.float32, copy=True)
        if self._vectors.ndim != 2 or self._vectors.shape[0] != len(self.item_ids):
            raise RetrievalError("Invalid dense vector dimensions")
        if self._vectors.shape[1] == 0 or not np.isfinite(self._vectors).all():
            raise RetrievalError("Invalid dense vectors")
        norms = np.linalg.norm(self._vectors.astype(np.float64), axis=1, keepdims=True)
        if (norms == 0).any():
            raise RetrievalError("Dense document vectors must not be zero")
        self._vectors /= norms
        self._encoder = encoder

    def search(
        self, text: str, limit: int = 10, excluded: Set[str] = frozenset()
    ) -> list[SearchHit]:
        text = _query(text, limit)
        if not text:
            return []
        query = np.asarray(self._encoder.encode([text]), dtype=np.float32)
        if query.shape != (1, self._vectors.shape[1]) or not np.isfinite(query).all():
            raise RetrievalError("Query encoder dimensions or values are invalid")
        norm = np.linalg.norm(query.astype(np.float64))
        if norm == 0:
            return []
        return _hits(self.item_ids, self._vectors @ (query[0] / norm), limit, excluded)


def choose_index(preferred: DenseIndex | None, fallback: TfidfIndex) -> DenseIndex | TfidfIndex:
    """Only an unavailable encoder triggers fallback; invalid artifacts must raise."""
    if preferred is not None and preferred.item_ids != fallback.item_ids:
        raise RetrievalError("Preferred and fallback indexes refer to different items")
    return fallback if preferred is None else preferred
