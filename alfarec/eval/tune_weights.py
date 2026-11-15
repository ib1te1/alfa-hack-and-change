"""Frozen tune-only weight search, blind supplements and a separate holdout check."""

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from alfarec.eval import judgments as j
from alfarec.eval.profiles import load_profiles
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH
from alfarec.recsys.ranking import DEFAULT_WEIGHTS, load_ranker
from alfarec.recsys.weights import DEFAULT_WEIGHTS_PATH, weight_artifact

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "weight_tuning"


def candidates() -> dict[str, dict[str, float]]:
    result = {"initial": dict(DEFAULT_WEIGHTS)}
    for feature in DEFAULT_WEIGHTS:
        if feature != "sequence":
            for factor in (0.5, 1.5):
                result[f"{feature}_{factor}"] = DEFAULT_WEIGHTS | {
                    feature: DEFAULT_WEIGHTS[feature] * factor
                }
    return result


def sources(encoder: str) -> dict[str, str]:
    paths = j.SOURCES | {
        "ranking_code": ROOT / "alfarec/recsys/ranking.py",
        "judgments_code": Path(j.__file__),
        "tuning_code": Path(__file__),
        "weight_code": ROOT / "alfarec/recsys/weights.py",
        "intent": ROOT / "alfarec/recsys/intent.yaml",
        "profile_code": ROOT / "alfarec/eval/profiles.py",
        "config_code": ROOT / "alfarec/config.py",
    }
    paths.update(
        {
            path.relative_to(ROOT).as_posix(): path
            for path in sorted((ROOT / "alfarec/recsys").glob("*.py"))
        }
    )
    if encoder == "dense":
        paths["ml_manifest"] = ROOT / "artifacts/ml/manifest.json"
    return {key: j.file_sha256(path) for key, path in paths.items()}


def collect_answers(
    pool: dict, paths: list[Path], resolutions: Path | None = None
) -> tuple[dict, list[dict], dict]:
    answers = [j.load_answers(path, pool) for path in paths]
    if sorted(a["labeler"] for a in answers) != ["A", "B"]:
        raise ValueError("Need exactly one independently confirmed answer from A and B")
    pairs = defaultdict(dict)
    for answer in answers:
        for row in answer["answers"]:
            pairs[(row["profile_id"], row["item_id"])][answer["labeler"]] = row["grade"]
    shared = [v for v in pairs.values() if len(v) == 2]
    numeric = [v for v in shared if v["A"] is not None and v["B"] is not None]
    conflicts = [
        {"profile_id": p, "item_id": i, **v}
        for (p, i), v in sorted(pairs.items())
        if len(v) == 2 and v["A"] != v["B"]
    ]
    # Disagreements remain provisional and block selecting weights below.
    grades = {
        key: min(numbers) if (numbers := [g for g in value.values() if g is not None]) else None
        for key, value in pairs.items()
    }
    agreement = {
        "shared_pairs": len(shared),
        "numeric_pairs": len(numeric),
        "kappa": j.cohen_kappa([v["A"] for v in numeric], [v["B"] for v in numeric]),
        "disagreements": conflicts,
    }
    if resolutions:
        resolved = json.loads(resolutions.read_text("utf-8"))
        if resolved.get("pool_sha256") != pool["pool_sha256"]:
            raise ValueError("Resolutions belong to a different pool")
        expected = {(r["profile_id"], r["item_id"]) for r in conflicts}
        seen = set()
        for row in resolved.get("answers", []):
            key = (row["profile_id"], row["item_id"])
            grade = row.get("grade")
            if (
                key not in expected
                or key in seen
                or (grade is not None and (type(grade) is not int or grade not in (0, 1, 2)))
                or not isinstance(row.get("note"), str)
                or not row["note"].strip()
            ):
                raise ValueError("Invalid or duplicate adjudication")
            seen.add(key)
            grades[key] = grade
        if seen != expected:
            raise ValueError("Every disagreement needs an explicit adjudication")
        agreement["resolutions_sha256"] = j.file_sha256(resolutions)
        conflicts = []
    return grades, conflicts, agreement


def context(profile: dict) -> dict:
    return {key: value for key, value in profile.items() if key != "items"}


def transferable(source: dict, target: dict, grades: dict, accept_instruction: bool) -> dict:
    if source["sources"]["labeling"] != target["sources"]["labeling"] and not accept_instruction:
        raise ValueError("Instruction changed; applicability must be explicitly confirmed")
    old = {p["id"]: p for p in source["profiles"]}
    new = {p["id"]: p for p in target["profiles"]}
    result = {}
    for (profile_id, item_id), grade in grades.items():
        if (
            profile_id in old
            and profile_id in new
            and context(old[profile_id]) == context(new[profile_id])
            and item_id in new[profile_id]["items"]
            and source["cards"].get(item_id) == target["cards"].get(item_id)
        ):
            result[(profile_id, item_id)] = grade
    return result


def freeze_pool(pool: dict) -> dict:
    pool.pop("pool_sha256", None)
    pool["pool_sha256"] = j.canonical_sha256(pool)
    return pool


def read_pool(path: Path) -> dict:
    pool = json.loads(path.read_text("utf-8"))
    if j.canonical_sha256({k: v for k, v in pool.items() if k != "pool_sha256"}) != pool.get(
        "pool_sha256"
    ):
        raise ValueError("Pool content does not match its hash")
    return pool


def grade_rows(grades: dict) -> list[dict]:
    return [
        {"profile_id": p, "item_id": i, "grade": grade} for (p, i), grade in sorted(grades.items())
    ]


def prepare(args) -> int:
    ranker = load_ranker(encoder_mode=args.encoder, weights_path=None)
    # Scenario histories must not depend on the model under evaluation.
    reference = load_ranker(encoder_mode="tfidf", weights_path=None)
    suite = load_profiles().profiles
    grid = candidates()
    lists, histories = {}, {}
    profiles, cards = [], {}
    for case in suite:
        history = j.resolve_history(case, reference) if case.history else []
        if history is None:
            raise ValueError(f"Unresolved fixed history: {case.id}")
        histories[case.id] = [event.model_dump(mode="json") for event in history]
        variants = {"baseline": j.top_list(ranker, case, history, "baseline")}
        for name, weights in grid.items():
            if case.split == "holdout" and name != "initial":
                continue
            candidate = copy.copy(ranker)
            candidate._weights = dict(weights)
            variants[name] = j.top_list(candidate, case, history, "ours")
        lists[case.id] = variants
        profile = j.observable_profile(case, history, ranker)
        profile["items"] = sorted({i for values in variants.values() for i in values})
        random.Random(f"{j.SEED}:{case.id}").shuffle(profile["items"])
        profiles.append(profile)
        for item_id in profile["items"]:
            cards[item_id] = j.public_card(ranker._items[item_id])
    shared, assigned = j.assignment(suite)
    pool = freeze_pool(
        {
            "version": 1,
            "seed": j.SEED,
            "sources": sources(args.encoder),
            "shared": shared,
            "assigned": assigned,
            "profiles": profiles,
            "cards": dict(sorted(cards.items())),
            "unresolved_history": [],
        }
    )
    transferred, agreement = {}, None
    if args.source_pool:
        source = read_pool(args.source_pool)
        grades, conflicts, agreement = collect_answers(
            source, args.source_answers, args.source_resolutions
        )
        if conflicts:
            raise ValueError("Resolve source disagreements before transferring ratings")
        transferred = transferable(source, pool, grades, args.accept_instruction_update)
    supplement = copy.deepcopy(pool)
    supplement["parent_pool_sha256"] = pool["pool_sha256"]
    for profile in supplement["profiles"]:
        profile["items"] = [i for i in profile["items"] if (profile["id"], i) not in transferred]
    freeze_pool(supplement)
    experiment = {
        "version": 1,
        "encoder": args.encoder,
        "sources": pool["sources"],
        "pool_sha256": pool["pool_sha256"],
        "candidates": grid,
        "lists": lists,
        "histories": histories,
        "transferred": grade_rows(transferred),
        "source_pool_sha256": source["pool_sha256"] if args.source_pool else None,
        "source_answers_sha256": {p.name: j.file_sha256(p) for p in args.source_answers},
        "instruction_update_accepted": args.accept_instruction_update,
        "source_agreement": agreement,
        "supplement_sha256": supplement["pool_sha256"],
    }
    experiment["experiment_sha256"] = j.canonical_sha256(experiment)
    for filename, payload in (
        ("pool.json", pool),
        ("supplement.json", supplement),
        ("experiment.json", experiment),
    ):
        j.write_json(args.output / filename, payload)
    (args.output / "form.html").write_text(j.render_form(supplement), "utf-8", newline="\n")
    print(
        json.dumps(
            {
                "pool": pool["pool_sha256"],
                "transferred": len(transferred),
                "missing": sum(len(p["items"]) for p in supplement["profiles"]),
                "profiles": len(suite),
                "candidates": len(grid),
            }
        )
    )
    return 0


def score(lists: dict[str, list[str]], grades: dict) -> dict:
    rows = []
    for profile_id, ranked in lists.items():
        numeric = {i: g for (p, i), g in grades.items() if p == profile_id and g is not None}
        rows.append(j.list_metrics(ranked, numeric))
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def choose(experiment: dict, grades: dict, splits: dict) -> tuple[str | None, dict]:
    # Neither candidate lists nor grades of holdout participate in this function.
    tune = {p: variants for p, variants in experiment["lists"].items() if splits[p] == "tune"}
    required = {(p, i) for p, variants in tune.items() for ids in variants.values() for i in ids}
    missing = sorted(required - grades.keys())
    if missing:
        return None, {"status": "missing_tune_judgments", "missing": missing}
    tune_grades = {key: grade for key, grade in grades.items() if key[0] in tune}
    metrics = {
        name: score({p: variants[name] for p, variants in tune.items()}, tune_grades)
        for name in experiment["candidates"]
    }
    # Fixed grid order resolves ties: initial first, then a single-feature change.
    winner = max(metrics, key=lambda name: metrics[name]["ndcg10"])
    return winner, {"status": "tune_selected", "winner": winner, "tune": metrics}


def select(args) -> int:
    experiment = json.loads((args.output / "experiment.json").read_text("utf-8"))
    if (
        j.canonical_sha256({k: v for k, v in experiment.items() if k != "experiment_sha256"})
        != (experiment["experiment_sha256"])
    ):
        raise ValueError("Experiment content does not match its hash")
    if experiment["sources"] != sources(experiment["encoder"]):
        raise ValueError("Experiment inputs changed; prepare a new blind pool")
    pool = read_pool(args.output / "pool.json")
    if pool["pool_sha256"] != experiment["pool_sha256"]:
        raise ValueError("Experiment and pool mismatch")
    grades = {(r["profile_id"], r["item_id"]): r["grade"] for r in experiment["transferred"]}
    if args.answers:
        supplement = read_pool(args.output / "supplement.json")
        if supplement["pool_sha256"] != experiment["supplement_sha256"]:
            raise ValueError("Experiment and supplement mismatch")
        extra, conflicts, agreement = collect_answers(supplement, args.answers, args.resolutions)
        j.write_json(args.output / "supplement_agreement.json", agreement)
        if conflicts:
            raise ValueError("Resolve supplement disagreements before selecting weights")
        grades.update(extra)
    splits = {p["id"]: p["split"] for p in pool["profiles"]}
    winner, report = choose(experiment, grades, splits)
    if winner:
        from alfarec.recsys.contracts import Event

        ranker = load_ranker(
            encoder_mode=experiment["encoder"],
            weights_path=None,
            weights=experiment["candidates"][winner],
        )
        holdout = {}
        for case in load_profiles().profiles:
            if case.split == "holdout":
                history = [Event.model_validate(e) for e in experiment["histories"][case.id]]
                holdout[case.id] = j.top_list(ranker, case, history, "ours")
        if args.holdout_answers:
            extra_pool = read_pool(args.output / "holdout_supplement.json")
            if extra_pool.get("parent_pool_sha256") != pool["pool_sha256"] or extra_pool.get(
                "candidate_sha256"
            ) != j.canonical_sha256(experiment["candidates"][winner]):
                raise ValueError("Holdout supplement belongs to a different selection")
            extra, conflicts, agreement = collect_answers(
                extra_pool, args.holdout_answers, args.holdout_resolutions
            )
            j.write_json(args.output / "holdout_agreement.json", agreement)
            if conflicts:
                raise ValueError("Resolve holdout disagreements before final evaluation")
            grades.update(extra)
        missing = sorted({(p, i) for p, ids in holdout.items() for i in ids} - grades.keys())
        report["holdout_lists"] = holdout
        if missing:
            report.update(status="missing_holdout_judgments", missing=missing)
            next_pool = copy.deepcopy(pool)
            next_pool["parent_pool_sha256"] = pool["pool_sha256"]
            next_pool["candidate_sha256"] = j.canonical_sha256(experiment["candidates"][winner])
            for _, item_id in missing:
                next_pool["cards"][item_id] = j.public_card(ranker._items[item_id])
            for profile in next_pool["profiles"]:
                profile["items"] = [i for p, i in missing if p == profile["id"]]
            freeze_pool(next_pool)
            j.write_json(args.output / "holdout_supplement.json", next_pool)
            (args.output / "holdout_form.html").write_text(j.render_form(next_pool), "utf-8")
        else:
            baseline = {p: experiment["lists"][p]["baseline"] for p in holdout}
            baseline_missing = {(p, i) for p, ids in baseline.items() for i in ids} - grades.keys()
            if baseline_missing:
                raise ValueError("Holdout baseline is not fully judged")
            report.update(
                status="complete",
                holdout={"ours": score(holdout, grades), "baseline": score(baseline, grades)},
            )
            report["experiment_sha256"] = experiment["experiment_sha256"]
            final_grades = grade_rows(grades)
            report["final_grades_sha256"] = j.canonical_sha256(final_grades)
            j.write_json(args.output / "final_grades.json", final_grades)
            report["evidence_sha256"] = [
                j.file_sha256(p)
                for p in (
                    *args.answers,
                    *args.holdout_answers,
                    *([args.resolutions] if args.resolutions else []),
                    *([args.holdout_resolutions] if args.holdout_resolutions else []),
                )
            ]
            artifact = weight_artifact(
                experiment["candidates"][winner],
                DEFAULT_CATALOG_PATH,
                {
                    "method": "fixed one-feature grid; tune only; ndcg10; initial wins ties",
                    "experiment_sha256": experiment["experiment_sha256"],
                    "winner": winner,
                    "encoder": experiment["encoder"],
                    "sources": experiment["sources"],
                    "evaluation_sha256": j.canonical_sha256(report),
                },
            )
            j.write_json(args.weights, artifact)
    j.write_json(args.output / "selection.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "missing": len(report.get("missing", [])),
                "winner": winner,
            }
        )
    )
    return 0 if report["status"] == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--encoder", choices=("dense", "tfidf"), default="dense")
    prepare_parser.add_argument("--source-pool", type=Path)
    prepare_parser.add_argument("--source-answers", type=Path, nargs=2, default=[])
    prepare_parser.add_argument("--accept-instruction-update", action="store_true")
    prepare_parser.add_argument("--source-resolutions", type=Path)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--answers", type=Path, nargs=2, default=[])
    select_parser.add_argument("--holdout-answers", type=Path, nargs=2, default=[])
    select_parser.add_argument("--resolutions", type=Path)
    select_parser.add_argument("--holdout-resolutions", type=Path)
    select_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    for command in (prepare_parser, select_parser):
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        return prepare(args) if args.command == "prepare" else select(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
