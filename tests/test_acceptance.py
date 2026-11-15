from pathlib import Path
from subprocess import CompletedProcess

import pytest

from alfarec.eval import acceptance
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


@pytest.mark.parametrize(
    "returncode,cases,expected",
    [
        (0, "<testcase/><testcase/>", True),
        (0, "<testcase/><testcase><skipped/></testcase>", None),
        (1, "<testcase/><testcase><failure/></testcase>", False),
        (0, "<testcase/>", False),
    ],
)
def test_paid_content_acceptance_requires_both_executed_tests(
    monkeypatch, returncode, cases, expected
):
    def run(command, **kwargs):
        report = next(
            arg.removeprefix("--junitxml=") for arg in command if arg.startswith("--junitxml=")
        )
        Path(report).write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
        return CompletedProcess(command, returncode, stdout="test summary", stderr="")

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert Checks.__new__(Checks).a07().passed is expected
