from presentation.build import card, holdout_table, markdown_tables, render, ru, verdict


def test_render_shows_holdout_metrics_and_removes_unmeasured_claims():
    rows = [
        ["Вариант", "nDCG@10", "Precision@5", "MRR"],
        ["Наша система", "0.499", "0.257", "0.333"],
        ["Бейзлайн", "0.250", "0.043", "0.054"],
    ]
    data = {name: {} for name in ("services", "services_baseline", "seller", "outside", "cold")}
    data["health"] = {"encoder": "dense", "recommender": "ranking", "catalog_items": 1497}
    data["seller_persona"] = {"title": "Демо", "legend": ""}
    result = render(data, {"`holdout` — основное сравнение, профилей: 14": rows}, {}, "191", False)
    assert "0,499" in result and "0,250" in result
    assert "holdout, 14 профилей" in result
    assert "nDCG@10 0,499 против 0,250 у бейзлайна — в 2,0 раза выше" in result
    assert "Ручная релевантность не измерена" not in result


def test_manual_relevance_uses_holdout_subheading_even_when_tune_is_first(tmp_path):
    report = tmp_path / "eval.md"
    report.write_text(
        "## Релевантность по ручной разметке\n"
        "### `tune`, профилей: 22\n"
        "| Вариант | nDCG@10 | Precision@5 | MRR |\n"
        "|---|---|---|---|\n"
        "| Наша система | 0.571 | 0.382 | 0.503 |\n"
        "### `holdout` — основное сравнение, профилей: 14\n"
        "| Вариант | nDCG@10 | Precision@5 | MRR |\n"
        "|---|---|---|---|\n"
        "| Наша система | 0.499 | 0.257 | 0.333 |\n"
        "| Бейзлайн | 0.250 | 0.043 | 0.054 |\n",
        encoding="utf-8",
    )
    count, rows = holdout_table(markdown_tables(report))
    assert count == "14"
    assert rows[1][1] == "0.499"
    assert rows[2][1] == "0.250"


def test_missing_manual_ratings_keep_the_unmeasured_state():
    assert holdout_table({"Следующий курс по поведению": [["Метод", "MRR"]]}) == ("?", [])


def test_numbers_use_decimal_comma_without_touching_identifiers():
    assert ru("e5-small 0.1303 / 12.60") == "e5-small 0,1303 / 12,60"
    assert ru("B04 (dense)") == "B04 (dense)"


def test_verdict_is_empty_without_a_baseline_row():
    assert verdict([["Вариант", "nDCG@10"], ["Наша система", "0.499"]]) == ""
    assert verdict([]) == ""


def test_card_keeps_the_history_reason_among_the_shown_two():
    entry = {
        "item": {"title": "Курс", "format": "course", "access": "free"},
        "reasons": [
            {"code": "task_match", "text": "Подходит под вашу задачу"},
            {"code": "industry_match", "text": "Для сферы «Услуги»"},
            {"code": "continues_history", "text": "По теме «Маркетплейсы» из вашей истории"},
        ],
        "next_step": {"label": "Начать курс"},
    }
    html = card(entry)
    assert "Подходит под вашу задачу" in html
    assert "<li class='hist'>По теме «Маркетплейсы» из вашей истории</li>" in html
    assert "Для сферы" not in html
