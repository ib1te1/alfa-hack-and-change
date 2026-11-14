import json
import sys

import pytest
from pydantic import ValidationError

from alfarec.eval import calibration as c
from alfarec.eval.profiles import load_profiles
from alfarec.recsys.catalog import load_catalog
from alfarec.recsys.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def packet():
    return c.build_packet()


def response_data(packet, rater="A"):
    # Synthetic software-test data only; never evidence of human calibration.
    return {
        "version": 1,
        "packet_sha256": packet.packet_sha256,
        "rater_id": rater,
        "independent": True,
        "instruction_feedback": "Synthetic test feedback",
        "answers": [
            {"pair_id": pair.pair_id, "score": score, "note": f"Synthetic test {pair.pair_id}"}
            for pair, score in zip(packet.pairs, [0, 1, 2, "unjudgeable", 1], strict=True)
        ],
    }


def test_committed_artifacts_match_current_sources(packet):
    saved = c.Packet.model_validate_json((c.DEFAULT_OUTPUT / "packet.json").read_text("utf-8"))
    assert saved == packet
    assert (c.DEFAULT_OUTPUT / "form.html").read_text("utf-8") == c.render_form(packet)


def test_pairs_are_real_tune_snapshots_without_hidden_fields(packet):
    profiles = {p.id: p for p in load_profiles().profiles}
    items = {i.item_id: i for i in load_catalog()}
    assert len({p.profile_id for p in packet.pairs}) == 5
    assert len({p.item.item_id for p in packet.pairs}) == 5
    for pair in packet.pairs:
        source = profiles[pair.profile_id]
        assert source.split == "tune" and not source.history
        assert pair.profile == source.profile
        assert pair.item.model_dump() == items[pair.item.item_id].model_dump(
            include=c.PublicCard.model_fields.keys()
        )
    serialized = packet.model_dump_json()
    for hidden in ("legend", "expected_behavior", "case", "split", "popularity", "raw_tags"):
        assert f'"{hidden}"' not in serialized
    assert '"score"' not in serialized


@pytest.mark.parametrize("profile_id", ["P02", "P19"])
def test_holdout_and_unresolved_history_are_rejected(profile_id):
    selections = c.Selections.model_validate_json(c.SELECTIONS.read_text("utf-8"))
    selections.pairs[0].profile_id = profile_id
    with pytest.raises(ValueError, match="tune profiles without history"):
        c.select_pairs(selections, load_profiles(), load_catalog(), load_taxonomy())


def test_missing_material_is_not_silently_replaced():
    selections = c.Selections.model_validate_json(c.SELECTIONS.read_text("utf-8"))
    selections.pairs[0].item_id = "missing"
    with pytest.raises(KeyError, match="missing"):
        c.select_pairs(selections, load_profiles(), load_catalog(), load_taxonomy())


@pytest.mark.parametrize("field", ["pair_id", "profile_id", "item_id"])
def test_duplicate_selections_are_rejected(field):
    values = json.loads(c.SELECTIONS.read_text("utf-8"))
    values["pairs"][1][field] = values["pairs"][0][field]
    with pytest.raises(ValidationError):
        c.Selections.model_validate(values)


def test_text_hash_ignores_git_line_endings(tmp_path):
    text = tmp_path / "text"
    text.write_bytes("Текст\nВторая строка\n".encode())
    original = c.file_hash(text)
    text.write_bytes("Текст\r\nВторая строка\r\n".encode())
    assert c.file_hash(text) == original
    assert c.file_hash(text, binary=True) != original


def test_presented_instructions_are_part_of_packet_version(packet, tmp_path, monkeypatch):
    template = tmp_path / "form.html"
    template.write_text(
        c.TEMPLATE.read_text("utf-8").replace("2 — прямо полезно", "New rubric"), encoding="utf-8"
    )
    monkeypatch.setattr(c, "TEMPLATE", template)
    changed = c.build_packet()
    assert changed.sources.form_template != packet.sources.form_template
    assert changed.packet_sha256 != packet.packet_sha256
    a = c.Response.model_validate(response_data(packet))
    b = c.Response.model_validate(response_data(changed, "B"))
    with pytest.raises(ValueError, match="different calibration packet"):
        c.compare(changed, a, b)


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 2])
def test_response_version_is_strict_integer_one(packet, version):
    values = response_data(packet)
    values["version"] = version
    with pytest.raises(ValidationError):
        c.Response.model_validate(values)


@pytest.mark.parametrize(
    "contents",
    [
        '{"score":2,"score":0}',
        '{"answers":[{"pair_id":"K01","score":2,"score":0}]}',
        '{"score":NaN}',
        '{"score":Infinity}',
        '{"score":-Infinity}',
    ],
)
def test_json_reader_rejects_duplicate_keys_and_nonfinite_constants(tmp_path, contents):
    path = tmp_path / "response.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        c.read_json(path)


def test_tampered_packet_is_rejected(packet):
    values = packet.model_dump()
    values["pairs"][0]["item"]["title"] = "Changed public card"
    with pytest.raises(ValidationError, match="checksum mismatch"):
        c.Packet.model_validate(values)


@pytest.mark.parametrize("score", [True, False, 1.0, 0.5, "1", None, -1, 3, "unknown"])
def test_invalid_scores_are_rejected(packet, score):
    values = response_data(packet)
    values["answers"][0]["score"] = score
    with pytest.raises(ValidationError):
        c.Response.model_validate(values)


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing", "blank_note", "blank_feedback", "extra"]
)
def test_incomplete_and_ambiguous_responses_are_rejected(packet, problem):
    values = response_data(packet)
    if problem == "duplicate":
        values["answers"][1]["pair_id"] = "K01"
    elif problem == "missing":
        values["answers"].pop()
    elif problem == "blank_note":
        values["answers"][0]["note"] = " \n "
    elif problem == "blank_feedback":
        values["instruction_feedback"] = "  "
    else:
        values["answers"][0]["expected_score"] = 2
    with pytest.raises(ValidationError):
        c.Response.model_validate(values)


@pytest.mark.parametrize("confirmation", [False, "true", 1])
def test_independence_requires_explicit_boolean_confirmation(packet, confirmation):
    values = response_data(packet)
    values["independent"] = confirmation
    with pytest.raises(ValidationError):
        c.Response.model_validate(values)


def test_comparison_preserves_original_scores_uncertainty_and_feedback(packet):
    a = c.Response.model_validate(response_data(packet))
    b_values = response_data(packet, "B")
    b_values["answers"][0]["score"] = "unjudgeable"
    b_values["answers"].reverse()
    b = c.Response.model_validate(b_values)
    report = c.compare(packet, a, b)
    assert report == c.compare(packet, b, a)
    assert "4 из 5" in report
    assert "A: **0**" in report and "B: **unjudgeable**" in report
    assert "Участник A: unjudgeable — 1 из 5" in report
    assert "Участник B: unjudgeable — 2 из 5" in report
    assert "Требуется разбор расхождения" in report
    assert report.count("Итоговая оценка после обсуждения: не зафиксирована") == 5
    assert "Synthetic test feedback" in report
    assert "B03 остаётся review" in report


@pytest.mark.parametrize("problem", ["same_rater", "wrong_packet"])
def test_comparison_rejects_same_rater_or_packet_mismatch(packet, problem):
    a = c.Response.model_validate(response_data(packet))
    values = response_data(packet, "A" if problem == "same_rater" else "B")
    if problem == "wrong_packet":
        values["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.compare(packet, a, c.Response.model_validate(values))


def test_source_strings_cannot_close_script_or_become_report_markup(packet):
    values = packet.model_dump(exclude={"packet_sha256"})
    values["pairs"][0]["item"]["title"] = '</script><img src=x onerror="alert(1)">'
    hostile = c.Packet.model_validate({**values, "packet_sha256": c.digest(values)})
    html = c.render_form(hostile)
    embedded = html.split('<script id="packet" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    assert "<" not in embedded
    assert json.loads(embedded)["pairs"][0]["item"]["title"] == hostile.pairs[0].item.title
    assert "innerHTML" not in html
    assert "localStorage" not in html and "fetch(" not in html
    assert c.quote("<script>\n[link](https://example.com)") == (
        "> &lt;script&gt;\n> \\[link\\]\\(https://example.com\\)"
    )


def test_cli_check_detects_stale_artifact_without_overwriting(packet, tmp_path, monkeypatch):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["calibration", "build", "--check", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        c.main()
    assert error.value.code == 2
    assert packet_path.read_text("utf-8") == "stale"


def test_cli_compare_does_not_overwrite_human_report(packet, tmp_path, monkeypatch):
    paths = []
    for rater in ("A", "B"):
        path = tmp_path / f"{rater}.json"
        path.write_text(json.dumps(response_data(packet, rater)), encoding="utf-8")
        paths.append(str(path))
    output = tmp_path / "review.md"
    monkeypatch.setattr(sys, "argv", ["calibration", "compare", *paths, "--output", str(output)])
    c.main()
    assert "не зафиксировано" in output.read_text("utf-8")
    output.write_text("Human adjudication", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        c.main()
    assert error.value.code == 2
    assert output.read_text("utf-8") == "Human adjudication"
