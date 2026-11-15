"""Build artifacts/course_transitions.json from the B02 enrolment handoff.

Usage: python -m alfarec.pipeline.build_transitions [--handoff DIR] [--output PATH]
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from alfarec.recsys.sequence import DEFAULT_TRANSITIONS_PATH, CourseTransitions

DEFAULT_HANDOFF = Path(__file__).resolve().parents[2] / "data" / "handoff"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRANSITIONS_PATH)
    args = parser.parse_args(argv)
    rows = pq.read_table(args.handoff / "course_enrolments.parquet").to_pylist()
    transitions = CourseTransitions.build(rows)
    transitions.save(args.output)
    pairs = sum(len(targets) for targets in transitions.counts.values())
    print(
        f"Course transitions: {pairs} pairs from {len(transitions.counts)} courses -> {args.output}"
    )


if __name__ == "__main__":
    main()
