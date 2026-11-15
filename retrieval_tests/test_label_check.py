import json
from types import SimpleNamespace

import pytest

from alfarec.eval import label_check
from alfarec.eval.label_check import f1_scores, field_scores, load_manual, stratified_sample


def test_sample_is_stratified_fixed_and_sized():
    items = [SimpleNamespace(item_id=f"a{n:03}", item_type="article") for n in range(300)]
    items += [SimpleNamespace(item_id=f"p{n}", item_type="podcast") for n in range(5)]
    first, second = stratified_sample(items, 40), stratified_sample(items, 40)
    assert first == second and len(first) == len(set(first)) == 40
    assert sum(i.startswith("p") for i in first) >= label_check.MIN_PER_TYPE


def test_f1_and_field_scores():
    assert f1_scores([({"a"}, {"a"}), ({"b"}, {"a"})], ["a", "b"]) == pytest.approx((2 / 3 + 0) / 2)
    vocab = {"levels": ["beginner", "advanced"], "stages": ["idea"], "tasks": ["t1", "t2"]}
    manual = {
        "x": {"level": "beginner", "stages": [], "tasks": ["t1"]},
        "y": {"level": None, "stages": ["idea"], "tasks": ["t2"]},
    }
    labels = {
        "x": {"level": "beginner", "stages": [], "tasks": ["t1", "t2"]},
        "y": {"level": "advanced", "stages": [], "tasks": ["t1"]},
    }
    scores = field_scores(manual, labels, vocab)
    assert scores["level"] == {"n": 1, "accuracy": 1.0, "macro_f1": 1.0}
    assert scores["stages"]["accuracy"] == 0.5  # both empty counts, missed idea does not
    assert scores["tasks"]["accuracy"] == 0.5


def test_answers_must_cover_the_committed_sample(tmp_path):
    sample = json.loads(label_check.SAMPLE_PATH.read_text(encoding="utf-8"))
    answers = [
        {"item_id": i["item_id"], "stages": [], "level": None, "tasks": [], "note": ""}
        for i in sample["items"]
    ]
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"sample_sha256": sample["sample_sha256"], "answers": answers}))
    assert len(load_manual(path, sample)) == 120
    path.write_text(json.dumps({"sample_sha256": sample["sample_sha256"], "answers": answers[1:]}))
    with pytest.raises(ValueError):
        load_manual(path, sample)


def test_committed_sample_is_current():
    assert label_check.sample_command(check=True) == 0
    form = label_check.FORM_PATH.read_text(encoding="utf-8")
    assert "/*SAMPLE*/null" not in form
