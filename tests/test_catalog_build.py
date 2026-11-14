from pathlib import Path

import pgdumplib
import pytest

from alfarec.pipeline.build_catalog import TABLES, build_catalog, write_index_text
from alfarec.pipeline.dump_reader import read_tables
from alfarec.pipeline.text import html_to_text, normalize_label, parse_duration, truncate
from alfarec.recsys.catalog import load_catalog, save_catalog

PUBLISHED = "status-published"
DRAFT = "status-draft"
LESSON_MARKER = "Закрытый текст урока для подписчиков"


def row(**values: object) -> dict[str, str | None]:
    return {key: None if value is None else str(value) for key, value in values.items()}


def article(item_id: str, **overrides: object) -> dict[str, str | None]:
    values = {
        "Id": item_id, "StatusId": PUBLISHED, "IsDeleted": "f", "IsClosed": "f", "IsPublic": "t",
        "Title": f"Статья {item_id}", "Description": "<p>Описание&nbsp;статьи</p>",
        "CardDescription": None, "Url": f"slug-{item_id}", "Published": "2026-01-02 10:00:00+00",
        "Duration": "7 минут", "SubjectId": "subject-ip", "ContentHeadings": "Шаг 1",
    }  # fmt: skip
    values.update(overrides)
    return row(**values)


def course(item_id: str, type_id: str, cost: object = None, **overrides: object) -> dict:
    values = {
        "Id": item_id, "StatusId": PUBLISHED, "IsDeleted": "f", "Title": f"Курс {item_id}",
        "Description": "Публичное описание курса", "Url": f"course-{item_id}",
        "Published": "2026-02-01 09:00:00.5+00", "Duration": "2,5 часа", "Cost": cost,
        "TypeId": type_id, "DescriptionAdditional": None, "Target": "Предпринимателям",
    }  # fmt: skip
    values.update(overrides)
    return row(**values)


def make_tables() -> dict[str, list[dict[str, str | None]]]:
    tables: dict[str, list[dict[str, str | None]]] = {name: [] for name in TABLES}
    tables["Statuses"] = [row(Id=PUBLISHED, Title="Опубликовано"), row(Id=DRAFT, Title="Черновик")]
    tables["ArticleSubjects"] = [row(Id="subject-ip", Title="ИП", IsDeleted="f")]
    tables["Articles"] = [
        article("a-free"),
        article("a-closed", IsClosed="t", Duration="3 минуты"),
        article("a-draft", StatusId=DRAFT),
        article("a-deleted", IsDeleted="t"),
    ]
    tables["ArticleEmbedItems"] = [
        row(Id="e2", ArticleId="a-free", HtmlContent="<p>Второй абзац</p>", Order=2),
        row(Id="e1", ArticleId="a-free", HtmlContent="<h2>Первый</h2>", Order=1),
        row(Id="e3", ArticleId="a-closed", HtmlContent="<p>Платное тело статьи</p>", Order=1),
    ]
    tables["CourseTypes"] = [
        row(Id="t-open", Title="Открытый"),
        row(Id="t-sub", Title="По подписке"),
        row(Id="t-paid", Title="Платный"),
    ]
    tables["Courses"] = [
        course("c-open", "t-open"),
        course("c-sub", "t-sub"),
        course("c-paid", "t-paid", cost=3990),
    ]
    tables["CourseSubscriptions"] = [row(SubscriptionId="sub", CourseId="c-sub")]
    tables["Subscription"] = [row(Id="sub", Title="Курс+", IsDeleted="f")]
    tables["SubscriptionTariffs"] = [
        row(Id="tar-1", SubscriptionId="sub", Cost=2990, DiscountCost=990, IsMain="t"),
        row(Id="tar-2", SubscriptionId="sub", Cost=990, DiscountCost=495, IsMain="f"),
    ]
    tables["Landings"] = [
        row(
            Id="l1", StatusId=PUBLISHED, IsDeleted="f", Url="course-c-paid",
            Description="Лендинг курса", LandingDescription="Что вы получите",
            InfoBindings='[{"Order": 1, "Title": "Кому подойдёт", "Description": "Селлерам"}]',
        )
    ]  # fmt: skip
    tables["CourseLessonBindings"] = [
        row(CourseId="c-open", LessonId="lesson-1", IsDeleted="f"),
        row(CourseId="c-paid", LessonId="lesson-2", IsDeleted="f"),
    ]
    tables["LessonEmbedItems"] = [row(LessonId="lesson-2", HtmlContent=LESSON_MARKER)]
    tables["CourseUserBindings"] = [
        row(CourseId="c-open", Status=2, IsDeleted="f"),
        row(CourseId="c-open", Status=1, IsDeleted="f"),
        row(CourseId="c-open", Status=0, IsDeleted="f"),
        row(CourseId="c-open", Status=2, IsDeleted="t"),
    ]
    tables["UserRatings"] = [
        row(EntityId="lesson-1", Value=5, IsDeleted="f"),
        row(EntityId="lesson-1", Value=4, IsDeleted="f"),
        row(EntityId="calc-1", Value=3, IsDeleted="f"),
        row(EntityId="calc-1", Value=0, IsDeleted="f"),
    ]
    tables["Glossary"] = [
        row(Id="g-1", StatusId=PUBLISHED, IsDeleted="f", Title="Сальдо", Description="Разница",
            Url="saldo", Published=None, Duration="2 vbyens"),
    ]  # fmt: skip
    tables["CalculatorPages"] = [
        row(Id="calc-1", StatusId=PUBLISHED, IsDeleted="f", Title="Калькулятор НДС",
            Description="Считает НДС", Url="nds", Published="2026-03-01 00:00:00+00"),
    ]  # fmt: skip
    tables["Topics"] = [
        row(Id="top-1", Title="Маркeтплейсы", IsDeleted="f"),
        row(Id="top-2", Title="Маркетплейсы", IsDeleted="f"),
        row(Id="top-3", Title="Удалённая", IsDeleted="t"),
    ]
    tables["Tags"] = [row(Id="tag-1", Title=" Постановка задач", IsDeleted="f")]
    tables["Industries"] = [
        row(Id="ind-1", Title="Актуальное 🔥 ", IsDeleted="f"),
        row(Id="ind-2", Title="HoReСa", IsDeleted="f"),
    ]
    tables["ContentSettings"] = [row(Id="cs-1", EntityId="a-free", IsDeleted="f")]
    tables["ContentSettingsTopics"] = [
        row(ContentSettingsId="cs-1", TopicId="top-2", Order=2),
        row(ContentSettingsId="cs-1", TopicId="top-1", Order=1),
        row(ContentSettingsId="cs-1", TopicId="top-3", Order=3),
    ]
    tables["ContentSettingsTags"] = [row(ContentSettingsId="cs-1", TagId="tag-1", Order=1)]
    tables["ContentSettingsIndustries"] = [
        row(ContentSettingsId="cs-1", IndustryId="ind-1", Order=1),
        row(ContentSettingsId="cs-1", IndustryId="ind-2", Order=2),
    ]
    return tables


@pytest.fixture
def built():
    items, index_rows = build_catalog(make_tables(), "https://example.test/")
    return {item.item_id: item for item in items}, {entry["item_id"]: entry for entry in index_rows}


def test_only_published_live_materials_are_included(built):
    items, _ = built
    assert set(items) == {"a-free", "a-closed", "c-open", "c-sub", "c-paid", "g-1", "calc-1"}


def test_access_and_price_follow_course_type_and_subscription(built):
    items, _ = built
    assert (items["c-open"].access, items["c-open"].price_rub) == ("free", None)
    assert (items["c-sub"].access, items["c-sub"].price_rub) == ("subscription", 990)
    assert (items["c-paid"].access, items["c-paid"].price_rub) == ("paid", 3990)
    assert (items["a-closed"].access, items["a-closed"].price_rub) == ("subscription", 990)


def test_closed_body_and_lessons_never_reach_the_index(built):
    items, index = built
    assert "Второй абзац" in index["a-free"]["index_text"]
    assert "Платное тело статьи" not in index["a-closed"]["index_text"]
    assert "Кому подойдёт" in index["c-paid"]["index_text"]
    texts = [entry["index_text"] for entry in index.values()]
    texts += [item.model_dump_json() for item in items.values()]
    assert not any(LESSON_MARKER in text for text in texts)


def test_card_fields_are_cleaned_and_derived(built):
    items, index = built
    free = items["a-free"]
    assert free.summary == "Описание статьи"
    assert str(free.url) == "https://example.test/articles/slug-a-free"
    assert free.format == "guide" and free.duration_min == 7
    assert items["a-closed"].format == "quick_answer"
    assert free.subject == "ИП"
    assert free.topics == ["Маркетплейсы"]
    assert free.raw_tags == ["Постановка задач"]
    assert free.raw_industries == ["Актуальное", "HoReCa"]
    assert index["a-free"]["index_text"].index("Первый") < index["a-free"]["index_text"].index(
        "Второй"
    )
    assert str(items["g-1"].url) == "https://example.test/vocabulary/saldo"
    assert items["g-1"].duration_min == 2 and items["g-1"].published_at is None
    assert items["c-paid"].duration_min == 150 and items["c-paid"].format == "course"
    assert items["calc-1"].format == "tool"


def test_engagement_uses_live_enrollments_and_valid_ratings(built):
    items, _ = built
    opened = items["c-open"]
    assert (opened.popularity, opened.completion_rate, opened.rating) == (3.0, 0.3333, 4.5)
    assert (items["calc-1"].popularity, items["calc-1"].rating) == (1.0, 3.0)
    assert items["c-paid"].popularity is None


def test_rebuild_is_byte_identical_and_round_trips(tmp_path: Path):
    outputs = []
    for run in range(2):
        items, index_rows = build_catalog(make_tables(), "https://example.test")
        catalog, index = tmp_path / f"catalog-{run}.parquet", tmp_path / f"index-{run}.parquet"
        save_catalog(items, catalog)
        write_index_text(index_rows, index)
        outputs.append((catalog.read_bytes(), index.read_bytes()))
    assert outputs[0] == outputs[1]
    assert load_catalog(tmp_path / "catalog-0.parquet") == items


def test_dump_reader_maps_copy_columns(tmp_path: Path):
    dump = pgdumplib.new("test")
    entry = dump.add_entry(
        "TABLE", namespace="public", tag="Topics",
        defn='CREATE TABLE public."Topics" ("Id" uuid, "Title" text);',
    )  # fmt: skip
    with dump.table_data_writer(entry, ['"Id"', '"Title"']) as writer:
        writer.append("top-1", "Финансы")
        writer.append("top-2", None)
    path = tmp_path / "test.backup"
    dump.save(path)
    assert read_tables(path, ["Topics"]) == {
        "Topics": [{"Id": "top-1", "Title": "Финансы"}, {"Id": "top-2", "Title": None}]
    }
    with pytest.raises(KeyError):
        read_tables(path, ["Missing"])


@pytest.mark.parametrize(
    ("raw", "minutes"),
    [
        ("7 минут", 7),
        ("2,5 часа", 150),
        ("1 час 25 минут", 85),
        ("~ 2 часа", 120),
        ("2 vbyens", 2),
        ("", None),
        (None, None),
        ("скоро", None),
    ],
)
def test_parse_duration(raw, minutes):
    assert parse_duration(raw) == minutes


def test_text_helpers():
    assert html_to_text("<p>Один</p><script>x()</script><p>Два&amp;три</p>") == "Один\nДва&три"
    assert truncate("слово " * 100, 30).endswith("…")
    assert len(truncate("слово " * 100, 30)) <= 30
    assert normalize_label("💥 NEW") == "NEW"
    assert normalize_label("Прoдажи") == "Продажи"


def test_mixed_script_text_is_fixed_per_letter_run():
    from alfarec.pipeline.text import fix_mixed_script

    assert fix_mixed_script("Маркетплейсы: с чего начать и как преуспeть") == (
        "Маркетплейсы: с чего начать и как преуспеть"
    )
    assert fix_mixed_script("CRM-системах и B2B-рынок") == "CRM-системах и B2B-рынок"
    assert fix_mixed_script("Самое важное o 115-ФЗ") == "Самое важное о 115-ФЗ"
    assert fix_mixed_script("план A") == "план A"
