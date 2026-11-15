"""Relevance judgments for B06 (spec section 9, docs/labeling.md): pool, form, merge, metrics.

Usage:
  python -m alfarec.eval.judgments pool [--check]
  python -m alfarec.eval.judgments merge A.json B.json

`pool` builds the blind pool of top-10 lists of our system, the baseline and three ablations for
the 36 test profiles, the assignment of profiles to labelers A and B (five profiles shared), and
a self-contained HTML form. `merge` validates two answer files, writes judgments.csv and the
agreement report with Cohen's kappa on the shared pairs. Metrics on the judgments are part of
`python -m alfarec.eval`.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path

from alfarec.eval.profiles import EvaluationProfile, load_profiles
from alfarec.recsys.contracts import Event
from alfarec.recsys.ranking import Ranker, load_ranker
from alfarec.recsys.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "artifacts" / "judgments"
POOL_PATH = POOL_DIR / "pool.json"
LISTS_PATH = POOL_DIR / "lists.json"
FORM_PATH = POOL_DIR / "form.html"
TEMPLATE_PATH = Path(__file__).with_name("judgments_form.html")
JUDGMENTS_PATH = Path(__file__).with_name("judgments.csv")
RESOLVED_PATH = Path(__file__).with_name("judgments_resolved.csv")
AGREEMENT_PATH = ROOT / "reports" / "judgments_agreement.md"
SEED = 20261115
K = 10
SHARED_TUNE, SHARED_HOLDOUT = 3, 2
VARIANTS = ("ours", "baseline", "no_stage_level", "no_history", "no_rerank")
# Team decision for the hackathon (docs/judgments.md): only the two compared systems are
# judged. Ablation lists are still built and kept, but their metrics are reported as not
# measured: scoring them on a partly judged pool would be biased.
POOLED = ("ours", "baseline")
VARIANT_LABELS = {
    "ours": "Наша система",
    "baseline": "Бейзлайн по теме",
    "no_stage_level": "Без стадии и уровня",
    "no_history": "Без истории",
    "no_rerank": "Без переранжирования",
}
# Manual check of resolved histories (docs/labeling.md): automatic choices that do not fit the
# selector are replaced by title. P24 rates a financing calculator; text search picked
# "Сумма прописью онлайн", which is not about financing.
HISTORY_OVERRIDES: dict[tuple[str, int], str] = {("P24", 0): "Калькулятор процентов"}
TIME_LABELS = {"quick": "5 минут", "hour": "около часа", "course": "готов к курсу"}
EVENT_LABELS = {
    "complete": "завершил",
    "enroll": "записался на",
    "save": "сохранил",
    "view": "открыл",
    "rate": "оценил",
    "known": "отметил «Уже знаю»",
    "dislike": "отметил «Не то»",
}
SOURCES = {
    "catalog": ROOT / "artifacts" / "catalog.parquet",
    "profiles": ROOT / "alfarec" / "eval" / "profiles.yaml",
    "taxonomy": ROOT / "alfarec" / "recsys" / "taxonomy.yaml",
    "transitions": ROOT / "artifacts" / "course_transitions.json",
    "index_manifest": ROOT / "artifacts" / "retrieval_tfidf" / "manifest.json",
    "labeling": ROOT / "docs" / "labeling.md",
}


def file_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".md", ".yaml", ".json", ".html", ".csv"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# History and lists ----------------------------------------------------------------------------


def resolve_history(case: EvaluationProfile, ranker: Ranker) -> list[Event] | None:
    """Turn history selectors into events: most popular matching item, or None if any fails."""
    items, facts = ranker._items, ranker._facts  # noqa: SLF001
    now = ranker._now  # noqa: SLF001
    used: set[str] = set()
    events = []
    for step in case.history:
        selector = step.selector
        matches = [
            item
            for item in items.values()
            if item.item_type == selector.item_type
            and item.item_id not in used
            and facts[item.item_id].tasks.get(selector.task, 0) >= 0.8
            and (selector.level is None or facts[item.item_id].level == selector.level)
        ]
        override = HISTORY_OVERRIDES.get((case.id, len(events)))
        if override:
            chosen = next(i for i in items.values() if i.title == override)
        elif matches:
            chosen = min(matches, key=lambda i: (-(i.popularity or 0), i.item_id))
        else:
            # Source labels miss some types (calculators have no tags): closest by text instead.
            task = load_taxonomy().tasks[selector.task]
            hits = ranker._index.search(f"{task.label} {task.description}", limit=300)  # noqa: SLF001
            found = [
                items[hit.item_id]
                for hit in hits
                if hit.item_id in items
                and hit.item_id not in used
                and items[hit.item_id].item_type == selector.item_type
                and (selector.level is None or facts[hit.item_id].level == selector.level)
            ]
            if not found:
                return None
            chosen = found[0]
        used.add(chosen.item_id)
        events.append(
            Event(
                user_id=f"eval-{case.id}",
                item_id=chosen.item_id,
                type=step.event,
                ts=now - timedelta(days=step.days_ago),
                value=step.value,
            )
        )
    return events


def variant_rankers(ranker: Ranker | None = None) -> dict[str, tuple[Ranker, str, bool]]:
    """Variant name -> (ranker, mode, whether history is passed)."""
    ours = ranker or load_ranker()
    return {
        "ours": (ours, "ours", True),
        "baseline": (ours, "baseline", True),
        "no_stage_level": (load_ranker(weights={"stage": 0.0, "level": 0.0}), "ours", True),
        "no_history": (ours, "ours", False),
        "no_rerank": (load_ranker(rerank=False), "ours", True),
    }


def top_list(ranker: Ranker, case: EvaluationProfile, history: list[Event], mode: str) -> list[str]:
    answer = ranker.recommend(case.profile, history, mode=mode, limit=K)
    cards = [entry.item.item_id for entry in answer.route + answer.more]
    return list(dict.fromkeys(cards))[:K]


def build_lists(
    variants: Mapping[str, tuple[Ranker, str, bool]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[Event]], list[str]]:
    reference = variants["ours"][0]
    lists: dict[str, dict[str, list[str]]] = {}
    histories: dict[str, list[Event]] = {}
    unresolved = []
    for case in load_profiles().profiles:
        history = resolve_history(case, reference) if case.history else []
        if history is None:
            unresolved.append(case.id)
            continue
        histories[case.id] = history
        lists[case.id] = {
            name: top_list(ranker, case, history if with_history else [], mode)
            for name, (ranker, mode, with_history) in variants.items()
        }
    return lists, histories, unresolved


# Pool and form --------------------------------------------------------------------------------


def observable_profile(case: EvaluationProfile, history: list[Event], ranker: Ranker) -> dict:
    """What a labeler sees: the profile fields and history, never the legend or expectations."""
    taxonomy = load_taxonomy()
    profile = case.profile
    fields = []
    if profile.stage:
        fields.append(["Стадия", taxonomy.stages[profile.stage].label])
    if profile.industries:
        names = [taxonomy.industries[i].label for i in profile.industries]
        fields.append(["Сфера", ", ".join(names)])
    if profile.tasks:
        fields.append(["Задачи", ", ".join(taxonomy.tasks[t].label for t in profile.tasks)])
    if profile.experience:
        fields.append(["Опыт", taxonomy.levels[profile.experience].label])
    if profile.time_budget:
        fields.append(["Время", TIME_LABELS[profile.time_budget]])
    if profile.free_text:
        fields.append(["Своими словами", profile.free_text])
    items = ranker._items  # noqa: SLF001
    past = [
        f"{event.ts:%d.%m.%Y}: {EVENT_LABELS.get(event.type, event.type)}"
        f" «{items[event.item_id].title}»"
        for event in history
    ]
    # docs/labeling.md: the legend explains the scenario but sets no hidden requirements.
    return {
        "id": case.id,
        "split": case.split,
        "legend": case.legend,
        "fields": fields,
        "history": past,
    }


def public_card(item) -> dict:
    return {
        "title": item.title,
        "summary": item.summary,
        "type": item.item_type,
        "format": item.format,
        "duration_min": item.duration_min,
        "access": item.access,
        "price_rub": item.price_rub,
    }


def assignment(cases: Sequence[EvaluationProfile]) -> tuple[list[str], dict[str, list[str]]]:
    rng = random.Random(SEED)
    tune = sorted(c.id for c in cases if c.split == "tune")
    holdout = sorted(c.id for c in cases if c.split == "holdout")
    shared = sorted(rng.sample(tune, SHARED_TUNE) + rng.sample(holdout, SHARED_HOLDOUT))
    rest = [c.id for c in sorted(cases, key=lambda c: c.id) if c.id not in shared]
    rng.shuffle(rest)
    own = {"A": sorted(rest[0::2]), "B": sorted(rest[1::2])}
    return shared, {code: sorted(own[code] + shared) for code in ("A", "B")}


def build_pool(ranker: Ranker | None = None) -> tuple[dict, dict]:
    variants = variant_rankers(ranker)
    reference = variants["ours"][0]
    lists, histories, unresolved = build_lists(variants)
    cases = [c for c in load_profiles().profiles if c.id in lists]
    shared, assigned = assignment(cases)
    items = reference._items  # noqa: SLF001
    profiles, cards = [], {}
    for case in cases:
        pooled = sorted({i for name in POOLED for i in lists[case.id][name]})
        random.Random(f"{SEED}:{case.id}").shuffle(pooled)
        entry = observable_profile(case, histories[case.id], reference)
        entry["items"] = pooled
        profiles.append(entry)
        for item_id in pooled:
            cards[item_id] = public_card(items[item_id])
    pool = {
        "version": 1,
        "seed": SEED,
        "sources": {name: file_sha256(path) for name, path in SOURCES.items()},
        "shared": shared,
        "assigned": assigned,
        "unresolved_history": unresolved,
        "profiles": profiles,
        "cards": dict(sorted(cards.items())),
    }
    # The ordered lists decide the metrics, so any change in them makes a new pool.
    pool["lists_sha256"] = canonical_sha256(lists)
    pool["pool_sha256"] = canonical_sha256(pool)
    return pool, {"pool_sha256": pool["pool_sha256"], "variants": list(VARIANTS), "lists": lists}


def render_form(pool: dict) -> str:
    data = json.dumps(pool, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("/*POOL*/null", data)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def pool_command(check: bool) -> int:
    pool, lists = build_pool()
    form = render_form(pool)
    if check:
        current = json.loads(POOL_PATH.read_text(encoding="utf-8")) if POOL_PATH.exists() else {}
        if current.get("pool_sha256") != pool["pool_sha256"]:
            print(
                "Judgment pool is out of date; rebuild it and collect new answers", file=sys.stderr
            )
            return 1
        print(f"Judgment pool is current: {pool['pool_sha256']}")
        return 0
    write_json(POOL_PATH, pool)
    write_json(LISTS_PATH, lists)
    FORM_PATH.write_text(form, encoding="utf-8", newline="\n")
    pairs = sum(len(p["items"]) for p in pool["profiles"])
    per = {code: sum(len(p["items"]) for p in pool["profiles"] if p["id"] in ids)
           for code, ids in pool["assigned"].items()}  # fmt: skip
    print(f"Pool: {len(pool['profiles'])} profiles, {pairs} pairs; A {per['A']}, B {per['B']}")
    print(f"Form: {FORM_PATH}")
    return 0


# Answers, agreement and metrics ---------------------------------------------------------------


def load_answers(path: Path, pool: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        raise ValueError(f"{path}: expected an object with a list of answers")
    errors = []
    if payload.get("version") != 1:
        errors.append("version must be 1")
    if payload.get("pool_sha256") != pool["pool_sha256"]:
        errors.append("answers were made for another pool")
    labeler = payload.get("labeler")
    if not isinstance(labeler, str) or labeler not in pool["assigned"]:
        errors.append("labeler must be A or B")
        labeler = None
    if payload.get("independent") is not True:
        errors.append("independence is not confirmed")
    expected = {
        (p["id"], item)
        for p in pool["profiles"]
        if p["id"] in pool["assigned"].get(labeler, [])
        for item in p["items"]
    }
    seen = set()
    for answer in payload["answers"]:
        if not isinstance(answer, dict) or not all(
            isinstance(answer.get(f), str) for f in ("profile_id", "item_id", "status", "note")
        ):
            errors.append("each answer needs string profile_id, item_id, status and note")
            continue
        key = (answer["profile_id"], answer["item_id"])
        status, grade = answer["status"], answer.get("grade")
        if key not in expected or key in seen:
            errors.append(f"unexpected or repeated pair {key}")
        seen.add(key)
        if status == "judged" and not (type(grade) is int and grade in (0, 1, 2)):
            errors.append(f"grade must be 0, 1 or 2 for {key}")
        if status == "unjudgeable" and (grade is not None or not answer["note"].strip()):
            errors.append(f"unjudgeable needs no grade and a note for {key}")
        if status not in {"judged", "unjudgeable"}:
            errors.append(f"unknown status for {key}")
    if missing := expected - seen:
        errors.append(f"{len(missing)} assigned pairs are not answered")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:10]))
    return payload


def cohen_kappa(left: Sequence[int], right: Sequence[int]) -> float | None:
    if not left:
        return None
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    ca, cb = Counter(left), Counter(right)
    expected = sum(ca[g] * cb[g] for g in (0, 1, 2)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def merge_command(paths: Sequence[Path]) -> int:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    answers = [load_answers(path, pool) for path in paths]
    if sorted(a["labeler"] for a in answers) != ["A", "B"]:
        print("Need one answer file from A and one from B", file=sys.stderr)
        return 1
    rows = []
    by_pair: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for payload in sorted(answers, key=lambda a: a["labeler"]):
        for answer in sorted(payload["answers"], key=lambda a: (a["profile_id"], a["item_id"])):
            rows.append([answer["profile_id"], answer["item_id"], payload["labeler"],
                         "" if answer["grade"] is None else str(answer["grade"]),
                         answer["status"], answer.get("note", "").replace("\n", " ")])  # fmt: skip
            by_pair[(answer["profile_id"], answer["item_id"])][payload["labeler"]] = answer
    import csv

    with JUDGMENTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["profile_id", "item_id", "labeler", "grade", "status", "note"])
        writer.writerows(rows)
    shared = {k: v for k, v in by_pair.items() if k[0] in pool["shared"]}
    numeric = [(v["A"]["grade"], v["B"]["grade"]) for v in shared.values()
               if v["A"]["grade"] is not None and v["B"]["grade"] is not None]  # fmt: skip
    kappa = cohen_kappa([a for a, _ in numeric], [b for _, b in numeric])
    differ = sorted(k for k, v in shared.items() if v["A"]["grade"] != v["B"]["grade"])
    lines = [
        "# Согласованность разметки B06",
        "",
        f"Пул `{pool['pool_sha256'][:12]}`, общие профили: {', '.join(pool['shared'])}."
        f" Общих пар с двумя числовыми оценками: {len(numeric)}."
        f" Каппа Коэна: {'не рассчитана' if kappa is None else f'{kappa:.3f}'}.",
        "",
        f"Расхождений: {len(differ)}. До разбора в метриках берётся меньшая оценка; итог"
        " разбора записывается в `alfarec/eval/judgments_resolved.csv`"
        " (`profile_id,item_id,grade`) и заменяет её.",
        "",
        "| Профиль | Материал | A | B |",
        "|---|---|---|---|",
    ]
    for profile_id, item_id in differ:
        pair = shared[(profile_id, item_id)]
        grade = lambda a: a["status"] if a["grade"] is None else str(a["grade"])  # noqa: E731
        lines.append(f"| {profile_id} | {item_id} | {grade(pair['A'])} | {grade(pair['B'])} |")
    AGREEMENT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Judgments: {JUDGMENTS_PATH}; agreement: {AGREEMENT_PATH}")
    return 0


def final_grades(path: Path = JUDGMENTS_PATH, resolved: Path = RESOLVED_PATH) -> dict:
    """(profile, item) -> grade; unjudgeable pairs are absent. Disagreements take the minimum."""
    import csv

    grades: dict[tuple[str, str], list[int]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "judged":
                grades[(row["profile_id"], row["item_id"])].append(int(row["grade"]))
    final = {key: min(values) for key, values in grades.items()}
    if resolved.exists():
        with resolved.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                final[(row["profile_id"], row["item_id"])] = int(row["grade"])
    return final


def list_metrics(ranked: Sequence[str], grades: Mapping[str, int]) -> dict[str, float | int]:
    """nDCG@10 with gains 2^g - 1; P@5 and MRR count grade 2 as relevant. Unjudged are skipped."""
    judged = [item for item in ranked if item in grades]
    gains = [2 ** grades[item] - 1 for item in judged[:K]]
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, 1))
    ideal_gains = sorted((2**g - 1 for g in grades.values()), reverse=True)[:K]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, 1))
    relevant = [grades[item] == 2 for item in judged]
    return {
        "ndcg10": dcg / idcg if idcg else 0.0,
        "p5": sum(relevant[:5]) / 5,
        "mrr": next((1 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0),
        "unjudged": len([item for item in ranked if item not in grades]),
    }


def evaluate_judgments(
    lists: Mapping[str, Mapping[str, Sequence[str]]],
    grades: Mapping[tuple[str, str], int],
    splits: Mapping[str, str],
    exclude: Callable[[str], bool] = lambda _: False,
) -> dict[str, dict[str, dict[str, float]]]:
    """split -> variant -> mean metrics over profiles that have any judged pair."""
    by_profile: dict[str, dict[str, int]] = defaultdict(dict)
    for (profile_id, item_id), grade in grades.items():
        by_profile[profile_id][item_id] = grade
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("tune", "holdout"):
        cases = [
            p for p in lists if splits.get(p) == split and by_profile.get(p) and not exclude(p)
        ]
        result[split] = {}
        for variant in POOLED:
            rows = [list_metrics(lists[p][variant], by_profile[p]) for p in cases]
            if rows:
                result[split][variant] = {
                    key: sum(r[key] for r in rows) / len(rows) for key in rows[0]
                } | {"profiles": len(rows)}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("pool", help="build the blind pool and the form")
    pool.add_argument("--check", action="store_true", help="only check the pool is current")
    merge = sub.add_parser("merge", help="merge answer files of labelers A and B")
    merge.add_argument("answers", nargs=2, type=Path)
    args = parser.parse_args(argv)
    if args.command == "pool":
        return pool_command(args.check)
    return merge_command(args.answers)


if __name__ == "__main__":
    raise SystemExit(main())
