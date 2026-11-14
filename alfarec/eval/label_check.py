"""Blind manual labels of 120 catalog items to check the B04 labels (spec 6.1, step 4).

Usage:
  python -m alfarec.eval.label_check sample [--check]
  python -m alfarec.eval.label_check score ANSWERS.json [--labels artifacts/item_labels.parquet]

`sample` draws 120 items (stratified by type, fixed seed) and writes a self-contained form; the
labeler sees only the public card. `score` compares the manual labels with item_labels.parquet
(B04) per field and per label source; without that file it scores the current rule labels
of the ranker, so the check works before B04.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from alfarec.recsys.catalog import load_catalog
from alfarec.recsys.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "label_check"
SAMPLE_PATH = OUT_DIR / "sample.json"
FORM_PATH = OUT_DIR / "form.html"
TEMPLATE_PATH = Path(__file__).with_name("label_check_form.html")
LABELS_PATH = ROOT / "artifacts" / "item_labels.parquet"
REPORT_PATH = ROOT / "reports" / "label_check.md"
SEED = 20261115
SIZE = 120
MIN_PER_TYPE = 3


def stratified_sample(items: Sequence, size: int = SIZE, seed: int = SEED) -> list[str]:
    """Proportional allocation by item type, at least MIN_PER_TYPE of each, fixed seed."""
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.item_id):
        by_type[item.item_type].append(item.item_id)
    quota = {
        t: max(MIN_PER_TYPE, round(size * len(ids) / len(items))) for t, ids in by_type.items()
    }
    # Trim the largest strata until the total matches the requested size.
    while sum(quota.values()) > size:
        largest = max(quota, key=lambda t: (quota[t], t))
        quota[largest] -= 1
    while sum(quota.values()) < size:
        largest = max(quota, key=lambda t: (len(by_type[t]) - quota[t], t))
        quota[largest] += 1
    chosen = [
        i for t in sorted(by_type) for i in rng.sample(by_type[t], min(quota[t], len(by_type[t])))
    ]
    rng.shuffle(chosen)
    return chosen


def build_sample() -> dict:
    catalog = {item.item_id: item for item in load_catalog()}
    taxonomy = load_taxonomy()
    ids = stratified_sample(list(catalog.values()))
    sample = {
        "version": 1,
        "seed": SEED,
        "catalog_sha256": hashlib.sha256(
            (ROOT / "artifacts" / "catalog.parquet").read_bytes()
        ).hexdigest(),
        "stages": {k: v.label for k, v in taxonomy.stages.items()},
        "levels": {k: v.label for k, v in taxonomy.levels.items()},
        "tasks": {k: v.label for k, v in taxonomy.tasks.items()},
        "items": [
            {
                "item_id": item_id,
                "title": catalog[item_id].title,
                "summary": catalog[item_id].summary,
                "type": catalog[item_id].item_type,
                "format": catalog[item_id].format,
                "duration_min": catalog[item_id].duration_min,
                "access": catalog[item_id].access,
            }
            for item_id in ids
        ],
    }
    text = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sample["sample_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return sample


def sample_command(check: bool) -> int:
    sample = build_sample()
    if check:
        current = (
            json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}
        )
        if current.get("sample_sha256") != sample["sample_sha256"]:
            print("Label sample is out of date", file=sys.stderr)
            return 1
        print(f"Label sample is current: {sample['sample_sha256']}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(
        json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    data = json.dumps(sample, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    form = TEMPLATE_PATH.read_text(encoding="utf-8").replace("/*SAMPLE*/null", data)
    FORM_PATH.write_text(form, encoding="utf-8", newline="\n")
    print(f"Sample: {len(sample['items'])} items; form: {FORM_PATH}")
    return 0


# Scoring --------------------------------------------------------------------------------------


def load_manual(path: Path, sample: dict) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sample_sha256") != sample["sample_sha256"]:
        raise ValueError("Answers were made for another sample")
    expected = {item["item_id"] for item in sample["items"]}
    answers = {a["item_id"]: a for a in payload.get("answers", [])}
    if set(answers) != expected:
        raise ValueError(f"Expected answers for all {len(expected)} items")
    for item_id, a in answers.items():
        if a["level"] is not None and a["level"] not in sample["levels"]:
            raise ValueError(f"Unknown level for {item_id}")
        if not set(a["stages"]) <= set(sample["stages"]) or not set(a["tasks"]) <= set(
            sample["tasks"]
        ):
            raise ValueError(f"Unknown stage or task for {item_id}")
        if len(a["tasks"]) > 3:
            raise ValueError(f"At most three tasks for {item_id}")
    return answers


def rule_labels() -> dict[str, dict]:
    """Current labels of the ranker (source labels and rules) in the item_labels shape."""
    from alfarec.recsys.ranking import load_ranker

    facts = load_ranker(transitions_path=None)._facts  # noqa: SLF001
    return {
        item_id: {
            "stages": sorted(f.stages),
            "level": f.level,
            "tasks": [t for t, s in sorted(f.tasks.items(), key=lambda kv: -kv[1]) if s >= 0.8][:3],
            "label_source": "rule",
        }
        for item_id, f in facts.items()
    }


def parquet_labels(path: Path) -> dict[str, dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return {
        r["item_id"]: {
            "stages": list(r.get("stages") or []),
            "level": r.get("level"),
            "tasks": list(r.get("tasks") or []),
            "label_source": r.get("label_source") or "unknown",
        }
        for r in rows
    }


def f1_scores(pairs: Iterable[tuple[set[str], set[str]]], labels: Iterable[str]) -> float:
    """Macro-F1 over labels (one-vs-rest); labels absent from both sides are skipped."""
    pairs = list(pairs)
    scores = []
    for label in labels:
        tp = sum(label in human and label in model for human, model in pairs)
        fp = sum(label not in human and label in model for human, model in pairs)
        fn = sum(label in human and label not in model for human, model in pairs)
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def field_scores(
    manual: Mapping[str, dict], labels: Mapping[str, dict], vocab: Mapping[str, Mapping]
) -> dict[str, dict[str, float]]:
    """Per field: accuracy (level: exact; sets: any overlap, both empty counts) and macro-F1."""
    ids = [i for i in manual if i in labels]
    result = {}
    level_ids = [i for i in ids if manual[i]["level"] is not None]
    if level_ids:
        pairs = [({manual[i]["level"]}, {labels[i]["level"]}) for i in level_ids]
        result["level"] = {
            "n": len(level_ids),
            "accuracy": sum(h == m for h, m in pairs) / len(pairs),
            "macro_f1": f1_scores(pairs, vocab["levels"]),
        }
    for field, key in (("stages", "stages"), ("tasks", "tasks")):
        pairs = [(set(manual[i][field]), set(labels[i][field])) for i in ids]
        hits = [(h & m) or (not h and not m) for h, m in pairs]
        result[field] = {
            "n": len(pairs),
            "accuracy": sum(bool(x) for x in hits) / len(pairs) if pairs else 0.0,
            "macro_f1": f1_scores(pairs, vocab[key]),
        }
    return result


def score_command(answers_path: Path, labels_path: Path) -> int:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    manual = load_manual(answers_path, sample)
    labels = parquet_labels(labels_path) if labels_path.exists() else rule_labels()
    origin = labels_path.name if labels_path.exists() else "текущие правила ранжирования"
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item_id in manual:
        if item_id in labels:
            by_source[labels[item_id]["label_source"]][item_id] = labels[item_id]
    lines = [
        "# Проверка разметки на 120 материалах",
        "",
        f"Выборка `{sample['sample_sha256'][:12]}`, ручные метки вслепую, сравниваются с:"
        f" {origin}. Уровень — точное совпадение; стадии и задачи — пересечение множеств"
        " (оба пустых — совпадение). Macro-F1 — по меткам один-против-остальных.",
        "",
        "| Источник | Поле | Материалов | Точность | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for source in ["все", *sorted(by_source)]:
        subset = labels if source == "все" else by_source[source]
        for field, m in field_scores(manual, subset, sample).items():
            lines.append(
                f"| {source} | {field} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |"
            )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Report: {REPORT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--check", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("answers", type=Path)
    score.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args(argv)
    if args.command == "sample":
        return sample_command(args.check)
    return score_command(args.answers, args.labels)


if __name__ == "__main__":
    raise SystemExit(main())
