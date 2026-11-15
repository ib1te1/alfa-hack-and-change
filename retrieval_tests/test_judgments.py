import json

import pytest

from alfarec.eval import judgments
from alfarec.eval.judgments import (
    assignment,
    cohen_kappa,
    evaluate_judgments,
    list_metrics,
    load_answers,
)
from alfarec.eval.profiles import load_profiles


def test_kappa_on_known_values():
    assert cohen_kappa([2, 1, 0, 2], [2, 1, 0, 2]) == 1.0
    assert cohen_kappa([], []) is None
    # Agreement no better than chance gives zero.
    assert cohen_kappa([2, 2, 0, 0], [2, 0, 2, 0]) == pytest.approx(0.0)


def test_list_metrics_skip_unjudged_and_use_graded_gains():
    grades = {"a": 2, "b": 0, "c": 1}
    perfect = list_metrics(["a", "c", "b"], grades)
    assert perfect["ndcg10"] == pytest.approx(1.0)
    assert perfect["mrr"] == 1.0 and perfect["p5"] == pytest.approx(0.2)
    worse = list_metrics(["x", "b", "c", "a"], grades)
    assert worse["unjudged"] == 1 and worse["mrr"] == pytest.approx(1 / 3)
    assert worse["ndcg10"] < 1.0


def test_assignment_shares_five_profiles_and_splits_the_rest():
    cases = load_profiles().profiles
    shared, assigned = assignment(cases)
    splits = {c.id: c.split for c in cases}
    assert len(shared) == 5 and {splits[p] for p in shared} == {"tune", "holdout"}
    own_a, own_b = set(assigned["A"]) - set(shared), set(assigned["B"]) - set(shared)
    assert own_a.isdisjoint(own_b)
    assert own_a | own_b | set(shared) == set(splits)


def pool_fixture() -> dict:
    return {
        "pool_sha256": "abc",
        "shared": ["P01"],
        "assigned": {"A": ["P01", "P02"], "B": ["P01"]},
        "profiles": [{"id": "P01", "items": ["i1", "i2"]}, {"id": "P02", "items": ["i3"]}],
    }


def answers(tmp_path, labeler="A", **change) -> object:
    payload = {
        "version": 1,
        "pool_sha256": "abc",
        "labeler": labeler,
        "independent": True,
        "answers": [
            {"profile_id": "P01", "item_id": "i1", "status": "judged", "grade": 2, "note": ""},
            {"profile_id": "P01", "item_id": "i2", "status": "unjudgeable", "grade": None,
             "note": "нет описания"},
            {"profile_id": "P02", "item_id": "i3", "status": "judged", "grade": 0, "note": ""},
        ],
    }  # fmt: skip
    payload.update(change)
    path = tmp_path / f"{labeler}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_answers_load(tmp_path):
    assert load_answers(answers(tmp_path), pool_fixture())["labeler"] == "A"


@pytest.mark.parametrize(
    "change",
    [
        {"pool_sha256": "other"},
        {"independent": False},
        {"labeler": "C"},
        {"answers": []},
        {"answers": [{"profile_id": "P01", "item_id": "i1", "status": "judged", "grade": 3}]},
    ],
)
def test_invalid_answers_are_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_answers(answers(tmp_path, **change), pool_fixture())


@pytest.mark.parametrize("note", [" ", None])
def test_unjudgeable_needs_a_note(tmp_path, note):
    payload = json.loads(answers(tmp_path).read_text(encoding="utf-8"))
    payload["answers"][1]["note"] = note
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_answers(path, pool_fixture())


@pytest.mark.parametrize("payload", [[], {"answers": [1]}, {"answers": [], "labeler": ["A"]}])
def test_malformed_files_raise_validation_errors(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_answers(path, pool_fixture())


def test_final_grades_take_the_minimum_until_resolved(tmp_path):
    csv = tmp_path / "j.csv"
    csv.write_text(
        "profile_id,item_id,labeler,grade,status,note\n"
        "P01,i1,A,2,judged,\nP01,i1,B,1,judged,\nP01,i2,A,,unjudgeable,x\n",
        encoding="utf-8",
    )
    assert judgments.final_grades(csv, tmp_path / "none.csv") == {("P01", "i1"): 1}
    resolved = tmp_path / "r.csv"
    resolved.write_text("profile_id,item_id,grade\nP01,i1,2\n", encoding="utf-8")
    assert judgments.final_grades(csv, resolved) == {("P01", "i1"): 2}


def test_metrics_are_reported_per_split():
    lists = {p: {v: ["i1", "i2"] for v in judgments.VARIANTS} for p in ("P01", "P02")}
    grades = {("P01", "i1"): 2, ("P02", "i2"): 1}
    result = evaluate_judgments(lists, grades, {"P01": "tune", "P02": "holdout"})
    assert result["tune"]["ours"]["profiles"] == 1
    assert result["holdout"]["baseline"]["ndcg10"] == pytest.approx(1.0)


def test_committed_reference_pool_is_current_and_hides_the_source(monkeypatch):
    # This is the pre-tuning TF-IDF reference. Final ML lists live in weight_tuning.
    from alfarec.recsys.ranking import load_ranker

    def reference_ranker(**kwargs):
        return load_ranker(weights_path=None, encoder_mode="tfidf", **kwargs)

    monkeypatch.setattr(judgments, "load_ranker", reference_ranker)
    pool = json.loads(judgments.POOL_PATH.read_text(encoding="utf-8"))
    rebuilt, lists = judgments.build_pool()
    assert rebuilt["pool_sha256"] == pool["pool_sha256"]
    assert rebuilt["lists_sha256"] == judgments.canonical_sha256(lists["lists"])
    committed_lists = json.loads(judgments.LISTS_PATH.read_text(encoding="utf-8"))
    assert committed_lists["lists"] == lists["lists"]
    text = json.dumps(pool, ensure_ascii=False)
    # The legend is shown as in docs/labeling.md; expectations and the source stay hidden.
    for hidden in ("expected_behavior", "score", "ours", "baseline"):
        assert f'"{hidden}"' not in text
    pooled = {p["id"]: set(p["items"]) for p in pool["profiles"]}
    for profile_id, variants in lists["lists"].items():
        judged = [set(variants[name]) for name in judgments.POOLED]
        assert set().union(*judged) == pooled[profile_id]
    form = judgments.FORM_PATH.read_text(encoding="utf-8")
    assert pool["pool_sha256"] in form and "/*POOL*/null" not in form
    assert set(lists["lists"]) == {p["id"] for p in pool["profiles"]}
