from alfarec.eval.acceptance import Checks, Result, render


def test_scenarios_without_subprocesses_pass_on_the_committed_artifacts():
    checks = Checks()
    for scenario in (checks.a04, checks.a05, checks.a09, checks.a10):
        result = scenario()
        assert result.passed, result.evidence


def test_report_marks_unchecked_and_failed_scenarios():
    text = render(
        [Result("A01", None, "нет Docker"), Result("A02", False, "x"), Result("A03", True, "y")],
        "abc1234",
    )
    assert "`abc1234`" in text
    assert "| A01 | не проверялся |" in text and "| A02 | **не пройден** |" in text
