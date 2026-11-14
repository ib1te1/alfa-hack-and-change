import pytest

from alfarec.recsys.contracts import Item, Profile, Recommender
from alfarec.recsys.stub import StubRecommender


def test_stub_is_deterministic_across_catalog_order(catalog: list[Item]) -> None:
    recommender: Recommender = StubRecommender(catalog)
    expected = recommender.recommend(Profile(), [], limit=5)
    assert expected == recommender.recommend(Profile(), [], limit=5)
    assert expected == StubRecommender(list(reversed(catalog))).recommend(Profile(), [], limit=5)
    assert len(expected.more) == 5
    assert len({entry.item.item_id for entry in expected.more}) == 5
    assert all(entry.score == 0 and entry.reasons == [] for entry in expected.more)
    assert expected.route == expected.quick == []


def test_stub_accepts_empty_catalog_and_small_catalog(catalog: list[Item]) -> None:
    assert StubRecommender([]).recommend(Profile(), []).more == []
    assert len(StubRecommender(catalog[:1]).recommend(Profile(), [], limit=10).more) == 1


def test_stub_does_not_share_mutable_items_or_profile(catalog: list[Item]) -> None:
    profile = Profile(tasks=["marketing"])
    stub = StubRecommender(catalog)
    expected = stub.recommend(profile, [])
    catalog[0].title = "Changed externally"
    response = stub.recommend(profile, [])
    assert response == expected
    response.more[0].item.title = "Changed response"
    response.profile_used.tasks.append("taxes")
    assert profile.tasks == ["marketing"]
    assert stub.recommend(profile, []) == expected


def test_stub_rejects_duplicate_ids(catalog: list[Item]) -> None:
    with pytest.raises(ValueError, match="unique"):
        StubRecommender([catalog[0], catalog[0]])


@pytest.mark.parametrize("limit", [0, 51, True, 1.5])
def test_stub_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="Limit"):
        StubRecommender([]).recommend(Profile(), [], limit=limit)


def test_stub_preserves_mode_but_does_not_claim_to_rank(catalog: list[Item]) -> None:
    response = StubRecommender(catalog).recommend(Profile(), [], mode="baseline")
    assert response.mode == "baseline"
    assert all(not entry.reasons for entry in response.more)
    with pytest.raises(ValueError, match="mode"):
        StubRecommender(catalog).recommend(Profile(), [], mode="invalid")
