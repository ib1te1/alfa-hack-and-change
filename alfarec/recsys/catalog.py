from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from alfarec.recsys.contracts import Item

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "catalog.parquet"

CATALOG_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("item_type", pa.string(), nullable=False),
        pa.field("format", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("summary", pa.string(), nullable=False),
        pa.field("duration_min", pa.int32()),
        pa.field("access", pa.string(), nullable=False),
        pa.field("price_rub", pa.int32()),
        pa.field("url", pa.string(), nullable=False),
        pa.field("topics", pa.list_(pa.string()), nullable=False),
        pa.field("raw_tags", pa.list_(pa.string()), nullable=False),
        pa.field("raw_industries", pa.list_(pa.string()), nullable=False),
        pa.field("subject", pa.string()),
        pa.field("published_at", pa.timestamp("us", tz="UTC")),
        pa.field("popularity", pa.float64()),
        pa.field("completion_rate", pa.float64()),
        pa.field("rating", pa.float64()),
    ]
)


def write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed writer options keep the file byte-identical across rebuilds of the same dump.
    pq.write_table(table, path, compression="zstd", write_statistics=False, store_schema=False)


def save_catalog(items: Sequence[Item], path: Path) -> None:
    rows = [item.model_dump(mode="python") for item in items]
    for row in rows:
        row["url"] = str(row["url"])
    write_parquet(pa.Table.from_pylist(rows, schema=CATALOG_SCHEMA), path)


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[Item]:
    """Read the public catalog artifact and validate every row against the contract."""
    table = pq.read_table(path, schema=CATALOG_SCHEMA)
    return [Item.model_validate(row) for row in table.to_pylist()]


def with_base_url(items: Sequence[Item], base_url: str) -> list[Item]:
    """Point material links to another platform host, keeping each page path."""
    base = base_url.rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"PLATFORM_BASE_URL must look like https://host, got {base_url!r}")
    return [
        Item.model_validate({**item.model_dump(), "url": base + urlsplit(str(item.url)).path})
        for item in items
    ]
