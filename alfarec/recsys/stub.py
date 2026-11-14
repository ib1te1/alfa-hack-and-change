import hashlib
from collections.abc import Sequence

from alfarec.recsys.contracts import (
    Event,
    Item,
    Mode,
    NextStep,
    Profile,
    Recommendation,
    RecommendationResponse,
)


class StubRecommender:
    def __init__(self, catalog: Sequence[Item]) -> None:
        if len({item.item_id for item in catalog}) != len(catalog):
            raise ValueError("Catalog item IDs must be unique")
        self._catalog = tuple(item.model_copy(deep=True) for item in catalog)

    def recommend(
        self,
        profile: Profile,
        history: Sequence[Event],
        mode: Mode = "ours",
        limit: int = 10,
    ) -> RecommendationResponse:
        if mode not in ("ours", "baseline"):
            raise ValueError("Unsupported recommendation mode")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Limit must be an integer from 1 to 50")

        # Stable sampling avoids process-dependent hashes and catalog-order effects.
        items = sorted(
            self._catalog,
            key=lambda item: (hashlib.sha256(item.item_id.encode()).digest(), item.item_id),
        )
        return RecommendationResponse(
            more=[
                Recommendation(
                    item=item.model_copy(deep=True),
                    score=0,
                    reasons=[],
                    next_step=NextStep(kind="read", label="Открыть материал", url=item.url),
                )
                for item in items[:limit]
            ],
            mode=mode,
            profile_used=profile.model_copy(deep=True),
        )
