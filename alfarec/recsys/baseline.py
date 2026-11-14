"""Topic baseline from section 6.6 of the spec: the reference the ranking is compared with."""

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
        # Keep the comparison baseline unchanged; the production fallback must still
        # honour explicit feedback and completed materials.
        excluded = (
            {event.item_id for event in history if event.type in {"complete", "known", "dislike"}}
            if mode == "ours"
            else set()
        )
        ranked = [(item, reasons) for item, reasons in self.ranked(profile)
                  if item.item_id not in excluded]  # fmt: skip
        return RecommendationResponse(
            more=[_recommendation(item, reasons) for item, reasons in ranked[:limit]],
            mode=mode,
            profile_used=profile.model_copy(deep=True),
        )


def _recommendation(item: Item, reasons: list[Reason]) -> Recommendation:
    return Recommendation(
        item=item.model_copy(deep=True),
        score=0.0,
        reasons=reasons,
        next_step=next_step_for(item),
    )
