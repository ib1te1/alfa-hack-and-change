from collections.abc import Iterable
from pathlib import Path

import pgdumplib

Row = dict[str, str | None]
Tables = dict[str, list[Row]]


def _columns(copy_stmt: str) -> list[str]:
    inner = copy_stmt.split("(", 1)[1].rsplit(")", 1)[0]
    return [column.strip().strip('"') for column in inner.split(",")]


def read_tables(path: Path, names: Iterable[str], namespace: str = "public") -> Tables:
    """Read table rows from a custom-format pg_dump without a PostgreSQL server.

    Values stay as the raw COPY text: booleans are "t"/"f", NULL is None.
    """
    dump = pgdumplib.load(path)
    columns = {
        entry.tag: _columns(entry.copy_stmt)
        for entry in dump.entries
        if entry.desc == "TABLE DATA" and entry.namespace == namespace
    }
    tables: Tables = {}
    for name in names:
        if name not in columns:
            raise KeyError(f"Table {namespace}.{name} is missing from {path}")
        tables[name] = [
            dict(zip(columns[name], row, strict=True)) for row in dump.table_data(namespace, name)
        ]
    return tables
