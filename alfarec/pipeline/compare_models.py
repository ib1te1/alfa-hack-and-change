"""GPU document encoding and CPU evaluation of B04 candidates, one process per model."""

import argparse
import gc
import json
import platform
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from alfarec.eval.offline import ndcg, recall
from alfarec.pipeline.build_index import prepare_texts
from alfarec.recsys.catalog import load_catalog
from alfarec.recsys.retrieval import TfidfIndex, digest, texts_digest, write_json

CANDIDATES = {
    "tiny": "cointegrated/rubert-tiny2",
    "e5-small": "intfloat/multilingual-e5-small",
    "user": "deepvk/USER-base",
    "e5-base": "intfloat/multilingual-e5-base",
}
REVISIONS = {
    "tiny": "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
    "e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "user": "e8446472f6024df155a04b2f0911b6044cabc51f",
    "e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a",
}
QUERIES = [
    "Как открыть бизнес",
    "Налоги и отчётность для начинающего предпринимателя",
    "Нужно привлечь клиентов в небольшую мастерскую с минимальным бюджетом",
    "Как выбрать налоговый режим для ИП, нанять первого сотрудника и подготовить договор",
    "Продажи",
    "Маркетплейсы и онлайн-продажи: первые шаги и расчёт затрат",
]


def evaluate(ids, query_vectors, vectors, source_ids, links):
    positions = {item: n for n, item in enumerate(ids)}
    scores = defaultdict(list)
    for source, query in zip(source_ids, query_vectors, strict=True):
        sim = np.asarray(vectors @ query).ravel()
        order = np.argsort(-sim, kind="stable")
        ranked = [ids[n] for n in order if n != positions[source]][:10]
        split = "tune" if zlib.crc32(source.encode()) % 10 < 6 else "holdout"
        scores[split + "_recall"].append(recall(ranked, links[source], 10))
        scores[split + "_ndcg"].append(ndcg(ranked, links[source], 10))
    return {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=["tfidf", *CANDIDATES], required=True)
    p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=Path("artifacts/catalog.parquet"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    output = args.output / args.candidate
    output.mkdir(parents=True, exist_ok=True)
    items = load_catalog(args.catalog)
    ids, texts = prepare_texts(
        items, source="index_text", index_text=args.handoff / "index_text.parquet"
    )
    by_id = {i.item_id: i for i in items}
    links = defaultdict(set)
    for row in pq.read_table(args.handoff / "editorial_pairs.parquet").to_pylist():
        if (
            row["source_id"] in by_id
            and row["item_id"] in by_id
            and row["source_id"] != row["item_id"]
        ):
            links[row["source_id"]].add(row["item_id"])
    sources = sorted(links)
    queries = [f"{by_id[i].title}. {by_id[i].summary}" for i in sources]
    manifest = {
        "candidate": args.candidate,
        "catalog_sha256": digest(args.catalog),
        "texts_sha256": texts_digest(ids, texts),
        "item_ids": ids,
        "editorial_sha256": digest(args.handoff / "editorial_pairs.parquet"),
        "cpu": platform.processor(),
        "threads": 4,
        "warmup": 5,
        "timed_queries": 30,
    }
    if args.candidate == "tfidf":
        index = TfidfIndex.fit(ids, texts)
        # Sparse multiplication avoids densifying the document vocabulary matrix.
        values = defaultdict(list)
        for source, query in zip(sources, queries, strict=True):
            ranked = [h.item_id for h in index.search(query, 10, {source})]
            split = "tune" if zlib.crc32(source.encode()) % 10 < 6 else "holdout"
            values[split + "_recall"].append(recall(ranked, links[source], 10))
            values[split + "_ndcg"].append(ndcg(ranked, links[source], 10))
        metrics = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in values.items()}

        def encode(text):
            return index.search(text, 10)

        index.save(
            output / "index",
            text_source="index_text",
            catalog_sha256=digest(args.catalog),
            texts_sha256=texts_digest(ids, texts),
        )
        manifest["weights_bytes"] = sum(
            f.stat().st_size for f in (output / "index").rglob("*") if f.is_file()
        )
    else:
        from huggingface_hub import snapshot_download

        from alfarec.recsys.encoder import SentenceEncoder

        model_id = CANDIDATES[args.candidate]
        revision = REVISIONS[args.candidate]
        path = snapshot_download(
            model_id,
            revision=revision,
            cache_dir=args.cache,
            token=False,
            ignore_patterns=["*.bin", "*.h5", "*.ot", "onnx/*", "openvino/*"],
        )
        encoder = SentenceEncoder(path, device="cuda", e5=args.candidate.startswith("e5"))
        vectors = encoder.documents(texts)
        query_vectors = encoder.encode(queries)
        metrics = evaluate(ids, query_vectors, vectors, sources, links)
        np.save(output / "item_vectors.npy", vectors.astype(np.float16), allow_pickle=False)
        manifest.update(
            model_id=model_id,
            revision=revision,
            model_path=str(Path(path).resolve()),
            e5=args.candidate.startswith("e5"),
            weights_bytes=sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()),
        )
        encoder.model.to("cpu")
        import torch

        torch.set_num_threads(4)
        torch.cuda.empty_cache()
        gc.collect()

        def encode(text):
            return encoder.encode([text])

    for n in range(5):
        encode(QUERIES[n % len(QUERIES)])
    elapsed = []
    for n in range(30):
        start = time.perf_counter()
        encode(QUERIES[n % len(QUERIES)])
        elapsed.append((time.perf_counter() - start) * 1000)
    manifest.update(
        metrics=metrics,
        cpu_p50_ms=float(np.percentile(elapsed, 50)),
        cpu_p95_ms=float(np.percentile(elapsed, 95)),
        queries=QUERIES,
    )
    if (output / "item_vectors.npy").exists():
        manifest["vectors_sha256"] = digest(output / "item_vectors.npy")
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"item_ids", "queries"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
