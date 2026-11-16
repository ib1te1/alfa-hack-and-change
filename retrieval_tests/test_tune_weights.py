import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alfarec.eval import tune_weights as t


def test_report_rejects_modified_weights_even_with_unchanged_provenance(tmp_path, monkeypatch):
    from alfarec.eval import offline

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "artifacts/weight_tuning"
    target.mkdir(parents=True)
    for filename in ("experiment.json", "selection.json", "final_grades.json"):
        (target / filename).write_bytes((root / "artifacts/weight_tuning" / filename).read_bytes())
    weights = json.loads((root / "artifacts/weights.json").read_text("utf-8"))
    weights["weights"]["history"] = 3.0
    t.j.write_json(tmp_path / "artifacts/weights.json", weights)
    monkeypatch.setattr(offline, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="stale or inconsistent"):
        offline.evaluate_judged()


def test_selected_weight_file_is_bound_to_complete_evaluation_and_final_grades(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    weights = json.loads((root / "artifacts/weights.json").read_text("utf-8"))
    folder = root / "artifacts/weight_tuning"
    selection = json.loads((folder / "selection.json").read_text("utf-8"))
    grades = json.loads((folder / "final_grades.json").read_text("utf-8"))
    assert selection["status"] == "complete"
    assert weights["provenance"]["evaluation_sha256"] == t.j.canonical_sha256(selection)
    assert selection["final_grades_sha256"] == t.j.canonical_sha256(grades)
    assert weights["weights"] == t.candidates()[selection["winner"]]
    from alfarec.eval.offline import evaluate_judged

    # This test checks the frozen evidence's integrity, independently of a local encoder.
    # Explanation-only source changes and list replay are covered below and in dense acceptance.
    pinned = json.loads((folder / "experiment.json").read_text("utf-8"))["sources"]
    monkeypatch.setattr(t, "sources", lambda _: pinned)
    reported = evaluate_judged()
    assert reported["holdout"]["ours"]["ndcg10"] == selection["holdout"]["ours"]["ndcg10"]


@pytest.mark.parametrize("changed", ["ranking_code", "ml_manifest", "profile_code"])
def test_judged_sources_reject_ranking_changes_without_loading_encoder(monkeypatch, changed):
    from alfarec.eval import offline

    pinned = {changed: "old"}
    monkeypatch.setattr(t, "sources", lambda _: {changed: "new"})
    # No profiles or candidates are needed: invalid sources must fail before model loading.
    assert not offline.judged_sources_match({"sources": pinned, "encoder": "dense"}, "winner")


@pytest.mark.parametrize("mismatch", [None, "baseline", "initial", "winner"])
def test_explanation_changes_require_exact_judged_lists(monkeypatch, mismatch):
    from alfarec.eval import offline

    source = "alfarec/recsys/explain.py"
    monkeypatch.setattr(t, "sources", lambda _: {source: "new"})
    data = {
        "sources": {source: "old"},
        "encoder": "dense",
        "histories": {"P01": []},
        "lists": {"P01": {name: [name, "second"] for name in ("baseline", "initial", "winner")}},
        "candidates": {name: {"name": name} for name in ("initial", "winner")},
    }
    monkeypatch.setattr(offline, "load_ranker", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        offline, "load_profiles", lambda: SimpleNamespace(profiles=[SimpleNamespace(id="P01")])
    )
    replayed = []

    def top_list(ranker, case, history, mode):
        assert case.id == "P01" and history == []
        name = "baseline" if mode == "baseline" else ranker._weights["name"]
        replayed.append(name)
        items = data["lists"][case.id][name]
        return list(reversed(items)) if name == mismatch else items

    monkeypatch.setattr(t.j, "top_list", top_list)
    assert offline.judged_sources_match(data, "winner") is (mismatch is None)
    if mismatch is None:
        assert set(replayed) == {"baseline", "initial", "winner"}


@pytest.mark.parametrize("folder", ["historical-652a8151", "supplement-9ebd99c2"])
def test_archived_raw_files_remain_unchanged_and_confirmation_only_changes_flag(folder):
    archive = Path(__file__).resolve().parents[1] / "reports/b06" / folder
    evidence = json.loads((archive / "validation.json").read_text("utf-8"))
    for code in ("A", "B"):
        raw_bytes = (archive / f"raw-{code}.json").read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == evidence["raw_sha256"][code]
        raw = json.loads(raw_bytes)
        confirmed = json.loads((archive / f"confirmed-{code}.json").read_text("utf-8"))
        assert raw["independent"] is False
        assert raw | {"independent": True} == confirmed


def experiment():
    return {
        "candidates": {"initial": {}, "task_1.5": {}},
        "lists": {
            "P01": {
                "initial": ["bad", "good"],
                "task_1.5": ["good", "bad"],
                "baseline": ["bad", "good"],
            },
            "P02": {"initial": ["holdout_item"]},
        },
    }


def test_selection_uses_tune_only_even_if_holdout_grades_change():
    splits = {"P01": "tune", "P02": "holdout"}
    grades = {("P01", "good"): 2, ("P01", "bad"): 0}
    first = t.choose(experiment(), grades, splits)
    second = t.choose(experiment(), grades | {("P02", "holdout_item"): 0}, splits)
    assert first == second
    assert first[0] == "task_1.5"


def test_any_missing_tune_pair_blocks_entire_search():
    winner, report = t.choose(experiment(), {("P01", "good"): 2}, {"P01": "tune", "P02": "holdout"})
    assert winner is None
    assert report["missing"] == [("P01", "bad")]


def test_unjudgeable_is_recorded_separately_from_missing():
    winner, report = t.choose(
        experiment(), {("P01", "good"): 2, ("P01", "bad"): None}, {"P01": "tune", "P02": "holdout"}
    )
    assert winner == "initial"  # equal lists after unjudgeable filtering; initial wins ties
    assert report["tune"]["initial"]["unjudged"] == 1


def test_grid_is_fixed_and_does_not_retune_course_transitions():
    grid = t.candidates()
    assert len(grid) == 23
    assert next(iter(grid)) == "initial"
    assert all(w["sequence"] == t.DEFAULT_WEIGHTS["sequence"] for w in grid.values())
    for weights in list(grid.values())[1:]:
        assert sum(weights[k] != grid["initial"][k] for k in weights) == 1


def test_transfer_requires_same_visible_context_card_and_instruction():
    pool = {
        "sources": {"labeling": "old"},
        "cards": {"x": {"title": "card"}},
        "profiles": [{"id": "P01", "fields": ["task"], "history": [], "items": ["x"]}],
    }
    target = copy.deepcopy(pool)
    grades = {("P01", "x"): 2}
    assert t.transferable(pool, target, grades, False) == grades
    target["sources"]["labeling"] = "new"
    with pytest.raises(ValueError, match="Instruction changed"):
        t.transferable(pool, target, grades, False)
    assert t.transferable(pool, target, grades, True) == grades
    target["profiles"][0]["history"] = ["different course"]
    assert not t.transferable(pool, target, grades, True)


def test_pool_digest_rejects_changed_content(tmp_path):
    pool = t.freeze_pool({"version": 1, "profiles": []})
    path = tmp_path / "pool.json"
    t.j.write_json(path, pool)
    assert t.read_pool(path) == pool
    pool["version"] = 2
    t.j.write_json(path, pool)
    with pytest.raises(ValueError, match="hash"):
        t.read_pool(path)


def test_cli_select_builds_new_holdout_cards_and_resumes_to_weight_artifact(tmp_path, monkeypatch):
    from alfarec.recsys.contracts import Item

    pool = t.freeze_pool(
        {
            "version": 1,
            "sources": {},
            "shared": [],
            "assigned": {"A": ["P01"], "B": ["P02"]},
            "cards": {"old": {"title": "Old"}},
            "profiles": [
                {"id": "P01", "split": "tune", "items": ["bad", "good"]},
                {"id": "P02", "split": "holdout", "items": ["old"]},
            ],
        }
    )
    experiment_data = experiment() | {
        "encoder": "tfidf",
        "sources": {},
        "pool_sha256": pool["pool_sha256"],
        "histories": {"P02": []},
        "transferred": t.grade_rows({("P01", "bad"): 0, ("P01", "good"): 2, ("P02", "old"): 1}),
    }
    experiment_data["candidates"] = {
        name: dict(t.DEFAULT_WEIGHTS) for name in experiment_data["candidates"]
    }
    experiment_data["lists"]["P02"] = {"initial": ["old"], "baseline": ["old"]}
    experiment_data["experiment_sha256"] = t.j.canonical_sha256(experiment_data)
    t.j.write_json(tmp_path / "pool.json", pool)
    t.j.write_json(tmp_path / "experiment.json", experiment_data)
    monkeypatch.setattr(t, "sources", lambda _: {})
    monkeypatch.setattr(
        t,
        "load_profiles",
        lambda: SimpleNamespace(profiles=[SimpleNamespace(id="P02", split="holdout")]),
    )
    new = Item(
        item_id="new",
        title="New card",
        item_type="article",
        format="guide",
        access="free",
        url="https://example.test/new",
    )
    monkeypatch.setattr(t, "load_ranker", lambda **_: SimpleNamespace(_items={"new": new}))
    monkeypatch.setattr(t.j, "top_list", lambda *args: ["new"])
    args = SimpleNamespace(
        output=tmp_path,
        answers=[],
        holdout_answers=[],
        resolutions=None,
        holdout_resolutions=None,
        weights=tmp_path / "weights.json",
    )
    assert t.select(args) == 2
    supplement = t.read_pool(tmp_path / "holdout_supplement.json")
    assert supplement["cards"]["new"]["title"] == "New card"
    assert not args.weights.exists()
    for code in ("A", "B"):
        path = tmp_path / f"{code}.json"
        t.j.write_json(
            path,
            {
                "version": 1,
                "pool_sha256": supplement["pool_sha256"],
                "labeler": code,
                "independent": True,
                "answers": []
                if code == "A"
                else [
                    {
                        "profile_id": "P02",
                        "item_id": "new",
                        "status": "judged",
                        "grade": 2,
                        "note": "",
                    }
                ],
            },
        )
        args.holdout_answers.append(path)
    assert t.select(args) == 0
    artifact = json.loads(args.weights.read_text("utf-8"))
    selection = json.loads((tmp_path / "selection.json").read_text("utf-8"))
    assert artifact["provenance"]["evaluation_sha256"] == t.j.canonical_sha256(selection)
    assert selection["final_grades_sha256"] == t.j.canonical_sha256(
        json.loads((tmp_path / "final_grades.json").read_text("utf-8"))
    )
