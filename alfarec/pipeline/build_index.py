import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.contracts import Item
from alfarec.recsys.retrieval import TfidfIndex, digest, texts_digest


def prepare_texts(
    items: list[Item], *, source: str, index_text: Path | None = None
) -> tuple[list[str], list[str]]:
    ordered = sorted(items, key=lambda item: item.item_id)
    ids = [item.item_id for item in ordered]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Catalog IDs must be nonempty and unique")
    if source == "metadata":
        return ids, [f"{item.title}\n{item.summary}" for item in ordered]
    if source != "index_text" or index_text is None:
        raise ValueError("index_text source requires an explicit text file")
    rows = pq.read_table(index_text).to_pylist()
    if any(set(row) != {"item_id", "item_type", "access", "index_text"} for row in rows):
        raise ValueError("Unexpected index_text schema")
    by_id = {row["item_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("Index text IDs must match the catalog exactly, without duplicates")
    for item in ordered:
        row = by_id[item.item_id]
        if (row["item_type"], row["access"]) != (item.item_type, item.access):
            raise ValueError(f"Text metadata differs from catalog: {item.item_id}")
        if not isinstance(row["index_text"], str) or not row["index_text"].strip():
            raise ValueError(f"Missing text for {item.item_id}")
    return ids, [by_id[item_id]["index_text"] for item_id in ids]


def build_index(
    catalog: Path, output: Path, *, source: str, index_text: Path | None = None
) -> TfidfIndex:
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {output}")
    catalog_hash = digest(catalog)
    ids, texts = prepare_texts(load_catalog(catalog), source=source, index_text=index_text)
    if digest(catalog) != catalog_hash:
        raise ValueError("Catalog changed while preparing the index")
    texts_hash = texts_digest(ids, texts)
    index = TfidfIndex.fit(ids, texts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".index-") as staging:
        staged = Path(staging) / "index"
        index.save(staged, text_source=source, catalog_sha256=catalog_hash, texts_sha256=texts_hash)
        TfidfIndex.load(
            staged,
            expected_item_ids=ids,
            expected_catalog_sha256=catalog_hash,
            expected_text_source=source,
            expected_texts_sha256=texts_hash,
        )
        staged.rename(output)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a reproducible offline TF-IDF index")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--source", choices=("index_text", "metadata"), default="index_text")
    parser.add_argument("--index-text", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        index = build_index(
            args.catalog, args.output, source=args.source, index_text=args.index_text
        )
    except (OSError, ValueError) as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1
    print(f"Built {index.mode}: {len(index.item_ids)} items, text_source={args.source}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
