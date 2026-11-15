"""Package selected B04 outputs and provenance; model weights stay outside Git."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from alfarec.recsys.retrieval import digest, write_json


def graphs(vectors, ids, enrolments):
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = vectors @ vectors.T
    similarities = np.clip(similarities, 0, 1)
    np.fill_diagonal(similarities, 0)
    rows, cols, values = [], [], []
    for n, scores in enumerate(similarities):
        for other in np.argsort(-scores, kind="stable")[:20]:
            if scores[other] > 0:
                rows.append(n)
                cols.append(other)
                values.append(float(scores[other]))
    semantic = sp.csr_matrix((values, (rows, cols)), shape=(len(ids), len(ids)))
    positions = {item: n for n, item in enumerate(ids)}
    users = defaultdict(set)
    for row in enrolments:
        if row["course_id"] in positions:
            users[row["user_key"]].add(positions[row["course_id"]])
    pairs = defaultdict(int)
    counts = defaultdict(int)
    for courses in users.values():
        for source in courses:
            counts[source] += 1
            for target in courses - {source}:
                pairs[source, target] += 1
    rows, cols, values = [], [], []
    for (source, target), count in sorted(pairs.items()):
        rows.append(source)
        cols.append(target)
        values.append(count / np.sqrt(counts[source] * counts[target]))
    behaviour = sp.csr_matrix((values, (rows, cols)), shape=semantic.shape)
    return semantic, behaviour


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/ml"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.model / "manifest.json").read_text("utf-8"))
    vectors = np.load(a.model / "item_vectors.npy", allow_pickle=False)
    semantic, behaviour = graphs(
        vectors,
        source["item_ids"],
        pq.read_table(a.handoff / "course_enrolments.parquet").to_pylist(),
    )
    shutil.copyfile(a.model / "item_vectors.npy", a.output / "item_vectors.npy")
    shutil.copyfile(a.labels, a.output / "item_labels.parquet")
    for name, graph in (
        ("semantic_graph.npz", semantic),
        ("behaviour_graph.npz", behaviour),
        ("item_graph.npz", semantic.maximum(behaviour)),
    ):
        sp.save_npz(a.output / name, graph)
    model_path = Path(source["model_path"])
    files = {
        f.relative_to(model_path).as_posix(): digest(f)
        for f in model_path.rglob("*")
        if f.is_file() and f.suffix in {".json", ".safetensors", ".txt", ".model"}
    }
    write_json(
        a.output / "manifest.json",
        {
            "version": 1,
            "catalog_sha256": source["catalog_sha256"],
            "item_ids": source["item_ids"],
            "model_id": source["model_id"],
            "revision": source["revision"],
            "e5": source["e5"],
            "model_files": files,
            "files": {
                name: digest(a.output / name)
                for name in [
                    "item_vectors.npy",
                    "item_labels.parquet",
                    "semantic_graph.npz",
                    "behaviour_graph.npz",
                    "item_graph.npz",
                ]
            },
            "graph_provenance": {
                "semantic": "top20 cosine of unit embeddings, no editorial edges",
                "behaviour": "symmetric co-enrolment cosine; full snapshot; "
                "prohibited in hidden-next-course evaluation",
                "enrolments_sha256": digest(a.handoff / "course_enrolments.parquet"),
            },
        },
    )


if __name__ == "__main__":
    main()
