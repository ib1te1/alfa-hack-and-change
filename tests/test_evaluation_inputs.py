from collections import Counter
from pathlib import Path

import pytest
import yaml

from alfarec.config import ConfigurationError
from alfarec.eval.profiles import load_profiles
from alfarec.recsys.taxonomy import load_taxonomy, match_key


def write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_bundled_profiles_cover_required_scenarios_and_taxonomy() -> None:
    taxonomy = load_taxonomy()
    suite = load_profiles(taxonomy=taxonomy)
    assert len(suite.profiles) == 36
    assert Counter(p.case for p in suite.profiles) == {
        "standard": 18,
        "history": 8,
        "minimal": 6,
        "out_of_scope": 4,
    }
    assert Counter(p.split for p in suite.profiles) == {"tune": 22, "holdout": 14}
    assert len({p.audience for p in suite.profiles}) == 3
    assert {task for p in suite.profiles for task in p.profile.tasks} == taxonomy.tasks.keys()
    assert {
        sphere for p in suite.profiles for sphere in p.profile.industries
    } == taxonomy.industries.keys()
    assert {p.profile.stage for p in suite.profiles if p.profile.stage} == taxonomy.stages.keys()
    for split in ("tune", "holdout"):
        part = [p for p in suite.profiles if p.split == split]
        assert {p.case for p in part} == {"standard", "history", "minimal", "out_of_scope"}
        assert len({p.audience for p in part}) == 3


@pytest.mark.parametrize(
    ("taxonomy_value", "source_value"),
    [
        ("Маркетплейсы", "Маркeтплейсы"),
        ("HoReCa", "HoReСa"),
        ("Финансы и учёт", "Финансы и учет"),
        ("Управление командой", "Управление командой\t"),
        ("ECom", "ecom"),
    ],
)
def test_source_spellings_share_a_match_key(taxonomy_value: str, source_value: str) -> None:
    assert match_key(taxonomy_value) == match_key(source_value)


def test_match_key_keeps_distinct_titles_apart() -> None:
    assert match_key("Продажи") != match_key("Продажи и переговоры")
    assert match_key("Финансы") != match_key("Финансы и учёт")


def test_default_configuration_paths_are_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert len(load_profiles().profiles) == 36


@pytest.mark.parametrize("bad_reference", ["tasks", "industries"])
def test_unknown_profile_references_are_reported(tmp_path: Path, bad_reference: str) -> None:
    payload = load_profiles().model_dump(mode="json")
    payload["profiles"][0]["profile"][bad_reference] = ["unknown_code"]
    with pytest.raises(ConfigurationError, match="P01.*unknown_code"):
        load_profiles(write_yaml(tmp_path, payload))


def test_unknown_history_task_is_reported(tmp_path: Path) -> None:
    payload = load_profiles().model_dump(mode="json")
    case = next(p for p in payload["profiles"] if p["case"] == "history")
    case["history"][0]["selector"]["task"] = "unknown_history_task"
    with pytest.raises(ConfigurationError, match="unknown_history_task"):
        load_profiles(write_yaml(tmp_path, payload))


def test_duplicate_profile_ids_are_rejected(tmp_path: Path) -> None:
    payload = load_profiles().model_dump(mode="json")
    payload["profiles"][1]["id"] = payload["profiles"][0]["id"]
    with pytest.raises(ConfigurationError, match="unique"):
        load_profiles(write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "document",
    ["version: 1\nversion: 1\n", "version: 1\ntasks:\n  tax: 1\n  tax: 2\n"],
)
def test_yaml_duplicate_keys_are_rejected(tmp_path: Path, document: str) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_taxonomy(path)


def test_yaml_object_construction_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="constructor"):
        load_taxonomy(path)


def test_missing_contract_vocabulary_is_rejected(tmp_path: Path) -> None:
    payload = load_taxonomy().model_dump(mode="json")
    del payload["stages"]["launch"]
    with pytest.raises(ConfigurationError, match="complete contract vocabulary"):
        load_taxonomy(write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "mutation", ["missing_history", "history_in_standard", "structured_minimal", "invalid_rating"]
)
def test_inconsistent_history_and_minimal_scenarios_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    payload = load_profiles().model_dump(mode="json")
    history = next(p for p in payload["profiles"] if p["case"] == "history")
    if mutation == "missing_history":
        history["history"] = []
    elif mutation == "history_in_standard":
        payload["profiles"][0]["history"] = history["history"]
    elif mutation == "structured_minimal":
        next(p for p in payload["profiles"] if p["case"] == "minimal")["profile"]["stage"] = "idea"
    else:
        history["history"][0].update(event="rate", value=True)
    with pytest.raises(ConfigurationError):
        load_profiles(write_yaml(tmp_path, payload))
