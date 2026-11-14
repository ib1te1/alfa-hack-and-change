"""Behavioural next-course signal: which course people enrol in after the ones a user took.

Built offline from course enrolments (B02 handoff) as aggregated course-to-course counts:
no user IDs, and pairs seen fewer than ``MIN_PAIR_COUNT`` times are dropped.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TRANSITIONS_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "course_transitions.json"
)
MIN_PAIR_COUNT = 2
# Chosen on the tune split of next-course cases (reports/eval.md): half transitions, half
# popularity, with each source course's counts smoothed toward popularity by five enrolments.
BLEND = 0.5
SMOOTHING = 5.0
FORMAT_VERSION = 1


@dataclass(frozen=True)
class CourseTransitions:
    counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @classmethod
    def build(cls, rows: Iterable[Mapping], min_count: int = MIN_PAIR_COUNT) -> "CourseTransitions":
        """Count ordered pairs (earlier course -> later course) within each user's enrolments."""
        by_user: dict[str, list[Mapping]] = defaultdict(list)
        for row in rows:
            if row.get("enrolled_at") is not None:
                by_user[row["user_key"]].append(row)
        pairs: Counter[tuple[str, str]] = Counter()
        for user in sorted(by_user):
            entries = sorted(by_user[user], key=lambda r: (r["enrolled_at"], r["course_id"]))
            seen: set[tuple[str, str]] = set()
            for i, earlier in enumerate(entries):
                for later in entries[i + 1 :]:
                    pair = (earlier["course_id"], later["course_id"])
                    # Same-month enrolments have no order; a user counts once per pair.
                    if pair[0] == pair[1] or later["enrolled_at"] == earlier["enrolled_at"]:
                        continue
                    if pair not in seen:
                        seen.add(pair)
                        pairs[pair] += 1
        counts: dict[str, dict[str, int]] = defaultdict(dict)
        for (source, target), count in sorted(pairs.items()):
            if count >= min_count:
                counts[source][target] = count
        return cls(dict(counts))

    def scores(
        self, history: Sequence[str], candidates: Mapping[str, float]
    ) -> dict[str, tuple[float, str]]:
        """Blend of transitions from the user's courses and popularity, scaled to max 1.

        ``candidates`` maps course IDs to their popularity; the result maps each to its score
        and the history course that contributes most (empty when only popularity counts).
        """
        sources = list(dict.fromkeys(history))
        total = sum(candidates.values())
        if not sources or total <= 0:
            return {}
        raw: dict[str, tuple[float, str]] = {}
        for course, popularity in candidates.items():
            share = popularity / total
            parts = []
            for source in sources:
                outgoing = self.counts.get(source, {})
                count = outgoing.get(course, 0)
                parts.append(((count + SMOOTHING * share) / (sum(outgoing.values()) + SMOOTHING),
                              count, source))  # fmt: skip
            followed = sum(p[0] for p in parts) / len(parts)
            best = max(parts, key=lambda p: (p[1], p[2]))
            raw[course] = (BLEND * followed + (1 - BLEND) * share, best[2] if best[1] else "")
        top = max(value for value, _ in raw.values())
        return (
            {course: (value / top, source) for course, (value, source) in raw.items()}
            if top
            else {}
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": FORMAT_VERSION, "min_pair_count": MIN_PAIR_COUNT,
                   "counts": self.counts}  # fmt: skip
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: Path) -> "CourseTransitions":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported course transitions format")
        return cls({source: dict(targets) for source, targets in payload["counts"].items()})
