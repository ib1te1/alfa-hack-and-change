import argparse
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from alfarec.pipeline.build_index import prepare_texts
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog
from alfarec.recsys.retrieval import TfidfIndex, digest, texts_digest, write_json
from alfarec.recsys.taxonomy import load_taxonomy


def benchmark(
    index_path: Path,
    catalog: Path,
    *,
    source: str,
    index_text: Path | None = None,
    repeats: int = 5,
) -> dict:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    ids, texts = prepare_texts(load_catalog(catalog), source=source, index_text=index_text)
    index = TfidfIndex.load(
        index_path,
        expected_item_ids=ids,
        expected_catalog_sha256=digest(catalog),
        expected_text_source=source,
        expected_texts_sha256=texts_digest(ids, texts),
    )
    taxonomy = load_taxonomy()
    queries = [entry.description for entry in [*taxonomy.tasks.values(), *taxonomy.stages.values()]]
    encode_ns, search_ns = [], []
    for query in queries:
        index.search(query, limit=200)
    for _ in range(repeats):
        for query in queries:
            started = perf_counter_ns()
            index.encode_query(query)
            encode_ns.append(perf_counter_ns() - started)
            started = perf_counter_ns()
            index.search(query, limit=200)
            search_ns.append(perf_counter_ns() - started)
    return {
        "mode": index.mode,
        "source": source,
        "catalog_sha256": digest(catalog),
        "texts_sha256": texts_digest(ids, texts),
        "index_manifest_sha256": digest(index_path / "manifest.json"),
        "items": len(ids),
        "features": index.provenance.features,
        "queries": len(queries),
        "samples": len(search_ns),
        "repeats": repeats,
        "limit": 200,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "versions": index.provenance.versions,
        "index_bytes": sum(path.stat().st_size for path in index_path.iterdir() if path.is_file()),
        "encode_p50_ms": float(np.percentile(encode_ns, 50) / 1_000_000),
        "encode_p95_ms": float(np.percentile(encode_ns, 95) / 1_000_000),
        "search_p50_ms": float(np.percentile(search_ns, 50) / 1_000_000),
        "search_p95_ms": float(np.percentile(search_ns, 95) / 1_000_000),
    }


def render_report(result: dict) -> str:
    scope = "публичных метаданных" if result["source"] == "metadata" else "переданном index_text"
    source_args = (
        "--source metadata"
        if result["source"] == "metadata"
        else "--source index_text --index-text data/processed/index_text.parquet"
    )
    return f"""# Замер индекса извлечения кандидатов B04

Промежуточный этап: `{result["mode"]}`, источник `{result["source"]}`. Проверена скорость
на {scope}; сравнение нейросетевых моделей и качество рекомендаций не измерены.
Запросы — описания 14 задач и 4 стадий таксономии, без оценок релевантности.

| Параметр | Значение |
|---|---|
| Материалов | {result["items"]} |
| Признаков | {result["features"]} |
| Запросов / повторов / замеров | {result["queries"]} / {result["repeats"]} / {result["samples"]} |
| Число кандидатов | {result["limit"]} |
| Размер индекса, байт | {result["index_bytes"]} |
| Кодирование p50 / p95, мс | {result["encode_p50_ms"]:.3f} / {result["encode_p95_ms"]:.3f} |
| Полный поиск p50 / p95, мс | {result["search_p50_ms"]:.3f} / {result["search_p95_ms"]:.3f} |
| Python | {result["python"]} |
| Среда | {result["platform"]} |
| Процессор | {result["processor"]} |

Версии библиотек: {result["versions"]}.

Каталог SHA-256: `{result["catalog_sha256"]}`.

Подготовленные тексты SHA-256: `{result["texts_sha256"]}`.

Манифест индекса SHA-256: `{result["index_manifest_sha256"]}`.

Перед измерением каждый запрос прогревается одним поиском. Используются perf_counter_ns
и процентили NumPy; время загрузки индекса, HTTP, дисковые операции и внешние модели
в измерение не входят. Полный поиск включает кодирование запроса, косинусную близость
и выбор кандидатов. Числа зависят от машины и нагрузки и не являются SLA сервиса.

Воспроизведение после установки requirements-retrieval.txt:

```sh
python -m alfarec.pipeline.build_index {source_args} --output data/processed/retrieval
python -m alfarec.eval.benchmark_retrieval {source_args} --index data/processed/retrieval
```

Для повторной сборки нужен новый каталог вывода: существующие артефакты не перезаписываются.
Пути каталога и текста должны соответствовать указанным контрольным суммам;
при другом расположении используйте --catalog и --index-text с фактическими путями.
Редакторские связи и оценки релевантности в этом замере не использованы: Recall/nDCG,
выбор нейросетевой модели, ручная оценка меток и граф поведения остаются в B04.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark offline retrieval on CPU")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--source", choices=("metadata", "index_text"), required=True)
    parser.add_argument("--index-text", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = benchmark(
            args.index,
            args.catalog,
            source=args.source,
            index_text=args.index_text,
            repeats=args.repeats,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_bytes(render_report(result).encode("utf-8"))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.json, result)
    except (OSError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(
        f"mode={result['mode']}, source={result['source']}, items={result['items']}; "
        f"encode_p95_ms={result['encode_p95_ms']:.3f}, search_p95_ms={result['search_p95_ms']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
