import pytest

from alfarec.recsys.contracts import Item


@pytest.fixture
def catalog() -> list[Item]:
    return [
        Item(
            item_id=f"item-{index}",
            item_type="article",
            format="guide",
            title=f"Материал {index}",
            access="free",
            url=f"https://example.com/articles/{index}",
        )
        for index in range(12)
    ]
