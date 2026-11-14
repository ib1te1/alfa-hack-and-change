import subprocess
import sys

import pytest

from alfarec.eval.sample_size import experiment_duration, sample_size_per_group


def test_independent_published_two_proportion_example() -> None:
    # Stata power twoproportions, example 1: 635 per group (iterative two-tail power).
    # The closed-form approximation differs by at most one after rounding.
    assert abs(sample_size_per_group(0.015, 0.001) - 635) <= 1


def test_sample_size_grows_when_effect_shrinks_or_power_increases() -> None:
    assert sample_size_per_group(0.01, 0.011) > sample_size_per_group(0.01, 0.012)
    assert sample_size_per_group(0.01, 0.012, power=0.9) > sample_size_per_group(0.01, 0.012)
    assert sample_size_per_group(0.01, 0.012, alpha=0.01) > sample_size_per_group(0.01, 0.012)


def test_group_order_does_not_change_required_sample_size() -> None:
    assert sample_size_per_group(0.01, 0.012) == sample_size_per_group(0.012, 0.01)


@pytest.mark.parametrize("bad", [0, 1, -0.01, float("nan"), float("inf"), True])
def test_invalid_conversion_inputs_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        sample_size_per_group(bad, 0.02)


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"power": 0.5}, {"power": float("inf")}])
def test_invalid_test_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        sample_size_per_group(0.01, 0.012, **kwargs)


def test_equal_probabilities_do_not_produce_a_finite_plan() -> None:
    with pytest.raises(ValueError, match="differ"):
        sample_size_per_group(0.01, 0.01)


def test_duration_includes_all_groups_enrollment_followup_and_data_lag() -> None:
    assert experiment_duration(43_000, 10_000) == (14, 44)
    assert experiment_duration(43_000, 10_000, traffic_fraction=0.1) == (86, 116)


@pytest.mark.parametrize(
    "kwargs", [{"traffic_fraction": 0}, {"traffic_fraction": 2}, {"data_lag_days": -1}]
)
def test_invalid_duration_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        experiment_duration(43_000, 10_000, **kwargs)


def test_cli_reports_missing_traffic_instead_of_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alfarec.eval.sample_size", "--daily-eligible", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "daily_eligible" in result.stderr
    assert "Traceback" not in result.stderr


def test_small_alpha_uses_a_representable_tail_probability() -> None:
    assert sample_size_per_group(0.01, 0.012, alpha=1e-17) > sample_size_per_group(0.01, 0.012)


def test_cli_reports_unrepresentable_duration_instead_of_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alfarec.eval.sample_size", "--traffic-fraction", "1e-310"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "numeric range" in result.stderr
    assert "Traceback" not in result.stderr
