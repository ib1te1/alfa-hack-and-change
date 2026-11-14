import argparse
import math
from statistics import NormalDist


def probability(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite probability strictly between 0 and 1")


def sample_size_per_group(
    control: float, treatment: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Normal approximation, two-sided test, equal groups, no continuity correction."""
    for name, value in (("control", control), ("treatment", treatment), ("alpha", alpha)):
        probability(value, name)
    if isinstance(power, bool) or not math.isfinite(power) or not 0.5 < power < 1:
        raise ValueError("Power must be a finite number strictly between 0.5 and 1")
    difference = abs(treatment - control)
    if difference == 0:
        raise ValueError("Control and treatment probabilities must differ")
    pooled = (control + treatment) / 2
    normal = NormalDist()
    if alpha / 2 == 0:
        raise ValueError("Alpha is too small to represent its two-sided tail")
    numerator = -normal.inv_cdf(alpha / 2) * math.sqrt(2 * pooled * (1 - pooled)) + normal.inv_cdf(
        power
    ) * math.sqrt(control * (1 - control) + treatment * (1 - treatment))
    ratio = numerator / difference
    estimate = ratio * ratio
    if not math.isfinite(estimate):
        raise ValueError("Required sample size exceeds the supported numeric range")
    return math.ceil(estimate)


def experiment_duration(
    per_group: int,
    daily_eligible: int,
    *,
    traffic_fraction: float = 1.0,
    followup_days: int = 28,
    data_lag_days: int = 2,
) -> tuple[int, int]:
    for name, value, minimum in (
        ("per_group", per_group, 1),
        ("daily_eligible", daily_eligible, 1),
        ("followup_days", followup_days, 1),
        ("data_lag_days", data_lag_days, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if (
        isinstance(traffic_fraction, bool)
        or not math.isfinite(traffic_fraction)
        or not 0 < traffic_fraction <= 1
    ):
        raise ValueError("Traffic fraction must be a finite number in (0, 1]")
    try:
        required_days = 2 * per_group / (daily_eligible * traffic_fraction)
    except OverflowError as error:
        raise ValueError("Required duration exceeds the supported numeric range") from error
    if not math.isfinite(required_days):
        raise ValueError("Required duration exceeds the supported numeric range")
    enrollment = max(14, math.ceil(required_days))
    return enrollment, enrollment + followup_days + data_lag_days


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a two-group conversion experiment")
    parser.add_argument("--baseline", type=float, default=0.01)
    parser.add_argument("--relative-lift", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--daily-eligible", type=int, default=10_000)
    parser.add_argument("--traffic-fraction", type=float, default=1.0)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.relative_lift) or args.relative_lift <= 0:
            raise ValueError("Relative lift must be finite and positive")
        per_group = sample_size_per_group(
            args.baseline,
            args.baseline * (1 + args.relative_lift),
            alpha=args.alpha,
            power=args.power,
        )
        enrollment, total = experiment_duration(
            per_group, args.daily_eligible, traffic_fraction=args.traffic_fraction
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"per_group={per_group}; total_users={2 * per_group}")
    print(f"enrollment_days={enrollment}; total_days_with_28d_followup_and_2d_lag={total}")
    print("Sensitivity: required users per group (alpha and power as requested)")
    print("baseline | +10% | +20% | +30%")
    for baseline in (0.005, 0.01, 0.02, 0.05):
        sizes = [
            sample_size_per_group(
                baseline, baseline * (1 + lift), alpha=args.alpha, power=args.power
            )
            for lift in (0.1, 0.2, 0.3)
        ]
        print(f"{baseline:.1%} | " + " | ".join(map(str, sizes)))


if __name__ == "__main__":
    main()
