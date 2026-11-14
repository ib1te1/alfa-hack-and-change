"""Topic baseline from section 6.6 of the spec and an interim ranking for the service.

The interim recommender only keeps the API and the page usable until the full ranking (B05)
replaces it: it reuses the baseline order, drops what the history marks as done or unwanted
and fills the quick block. It does not claim stage, level or industry fit.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

from alfarec.recsys.contracts import (
    Event,
    Item,
    Mode,
    Profile,
    Reason,
    ReasonCode,
    Recommendation,
    RecommendationResponse,
)
from alfarec.recsys.explain import next_step_for
from alfarec.recsys.taxonomy import Taxonomy, match_key

QUICK_FORMATS = {"quick_answer", "tool"}
QUICK_MAX_MIN = 5
QUICK_LIMIT = 4
EXCLUDING_EVENTS = {"complete", "known", "dislike"}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _popularity_order(item: Item) -> tuple:
    published = item.published_at or _EPOCH
    return (-(item.popularity or 0.0), -published.timestamp(), item.item_id)


class TopicBaseline:
    """Profile tasks map to raw topics; matching items sort by popularity, then by date."""

    def __init__(self, catalog: Sequence[Item], taxonomy: Taxonomy) -> None:
        if len({item.item_id for item in catalog}) != len(catalog):
            raise ValueError("Catalog item IDs must be unique")
        self._catalog = tuple(sorted(catalog, key=_popularity_order))
        self._taxonomy = taxonomy

    def ranked(self, profile: Profile) -> list[tuple[Item, list[Reason]]]:
        topic_tasks: dict[str, str] = {}
        for task_id in profile.tasks:
            task = self._taxonomy.tasks.get(task_id)
            if task:
                for topic in task.source_links.topics:
                    topic_tasks.setdefault(match_key(topic), task.label)
        if not topic_tasks:
            return [(item, []) for item in self._catalog]
        result = []
        for item in self._catalog:
            matched = [
                topic_tasks[key] for key in map(match_key, item.topics) if key in topic_tasks
            ]
            if matched:
                reason = Reason(code=ReasonCode.TASK_MATCH, weight=1.0, params={"task": matched[0]})
                result.append((item, [reason]))
        return result

    def recommend(
        self,
        profile: Profile,
        history: Sequence[Event],
        mode: Mode = "baseline",
        limit: int = 10,
    ) -> RecommendationResponse:
        return RecommendationResponse(
            more=[_recommendation(item, reasons) for item, reasons in self.ranked(profile)[:limit]],
            mode=mode,
            profile_used=profile.model_copy(deep=True),
        )


class InterimRecommender:
    """Serves mode="ours" until B05: baseline order with history and time-budget filters."""

    def __init__(self, catalog: Sequence[Item], taxonomy: Taxonomy) -> None:
        self._baseline = TopicBaseline(catalog, taxonomy)

    def recommend(
        self,
        profile: Profile,
        history: Sequence[Event],
        mode: Mode = "ours",
        limit: int = 10,
    ) -> RecommendationResponse:
        if mode == "baseline":
            return self._baseline.recommend(profile, history, mode, limit)
        excluded = {event.item_id for event in history if event.type in EXCLUDING_EVENTS}
        ranked = [(item, reasons) for item, reasons in self._baseline.ranked(profile)]
        ranked = [(item, reasons) for item, reasons in ranked if item.item_id not in excluded]
        quick = [
            (item, [*reasons, _short(item)])
            for item, reasons in ranked
            if item.format in QUICK_FORMATS
            and item.duration_min is not None
            and item.duration_min <= QUICK_MAX_MIN
        ][:QUICK_LIMIT]
        if profile.time_budget == "quick":
            ranked = [(item, reasons) for item, reasons in ranked if item.format in QUICK_FORMATS]
        elif profile.time_budget == "course":
            courses = [entry for entry in ranked if entry[0].format == "course"]
            ranked = courses + [entry for entry in ranked if entry[0].format != "course"]
        return RecommendationResponse(
            more=[_recommendation(item, reasons) for item, reasons in ranked[:limit]],
            quick=[_recommendation(item, reasons) for item, reasons in quick],
            mode=mode,
            profile_used=profile.model_copy(deep=True),
        )


def _short(item: Item) -> Reason:
    return Reason(code=ReasonCode.SHORT_FORMAT, weight=0.0, params={"minutes": item.duration_min})


def _recommendation(item: Item, reasons: list[Reason]) -> Recommendation:
    return Recommendation(
        item=item.model_copy(deep=True),
        score=0.0,
        reasons=reasons,
        next_step=next_step_for(item),
    )
