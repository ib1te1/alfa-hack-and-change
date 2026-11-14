from pydantic import Field

from alfarec.recsys.contracts import (
    Contract,
    Identifier,
    Mode,
    Profile,
)


class RecommendationRequest(Contract):
    profile: Profile = Field(default_factory=Profile)
    user_id: Identifier | None = None
    mode: Mode = "ours"
    limit: int = Field(default=10, ge=1, le=50, strict=True)
