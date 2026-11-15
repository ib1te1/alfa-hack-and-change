"""Build the B12 presentation: HTML slides from live data, then a PDF through headless Edge.

Usage: python presentation/build.py [--browser PATH] [--skip-pdf]

Metrics come from reports/eval.md and reports/b04_ml.md, examples from the running API
(in-process), and the page screenshot from a local server; nothing is typed in by hand, so the
deck matches the commit. Examples follow the encoder the local service actually loads: build the
submission PDF with the packaged model installed (see README), otherwise they show the fallback.
Output goes to dist/ (outside git): presentation.html, page.png, alfa-course-recommendations.pdf.
"""

import argparse
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from alfarec.api.main import create_app  # noqa: E402

DIST = ROOT / "dist"
REPO_URL = "https://github.com/ib1te1/alfa-hack-and-change"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/microsoft-edge",
]
FORMAT_LABELS = {
    "quick_answer": "Короткий ответ",
    "guide": "Руководство",
    "template": "Шаблон",
    "tool": "Калькулятор",
    "course": "Курс",
    "recording": "Запись вебинара",
    "podcast": "Подкаст",
}
ACCESS_LABELS = {"free": "бесплатно", "subscription": "по подписке «Курс+»", "paid": "платно"}

e = html.escape


# Data ----------------------------------------------------------------------------------------


def markdown_tables(path: Path) -> dict[str, list[list[str]]]:
    """Tables of a Markdown report keyed by the heading above them (first table wins)."""
    tables: dict[str, list[list[str]]] = {}
    heading = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif line.startswith("|") and not re.fullmatch(r"\|[-| :]+\|", line.strip()):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            tables.setdefault(heading, []).append(cells)
    return tables


def case_count(path: Path) -> str:
    match = re.search(r"Случаев в `holdout`[^:]*: (\d+)", path.read_text(encoding="utf-8"))
    return match.group(1) if match else "?"


def holdout_table(tables: dict[str, list[list[str]]]) -> tuple[str, list[list[str]]]:
    """Choose the held-out relevance table, not a parent heading or the tune table."""
    for heading, rows in tables.items():
        if re.match(r"^`?holdout`?(?:\s|$)", heading) and rows and "nDCG@10" in rows[0]:
            match = re.search(r"профилей:\s*(\d+)", heading)
            count = match.group(1) if match else "?"
            return count, rows
    return "?", tables.get("Релевантность по ручной разметке", [])


def commit_label() -> str:
    """Short SHA of the commit the deck is built on, flagged when tracked files differ from it."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return ""
    return f"{sha} + незакоммиченные изменения" if dirty else sha


def packaged_model() -> str:
    """Encoder the ML artifacts were built with, empty when they are absent."""
    manifest = ROOT / "artifacts" / "ml" / "manifest.json"
    if not manifest.is_file():
        return ""
    return str(json.loads(manifest.read_text(encoding="utf-8")).get("model_id", ""))


def columns(rows: list[list[str]], keep: list[int]) -> list[list[str]]:
    return [[row[i] for i in keep] for row in rows]


def row_index(rows: list[list[str]], name: str) -> int | None:
    """Position of a body row by its first cell, for table(highlight=...)."""
    for index, row in enumerate(rows[1:]):
        if row[0] == name:
            return index
    return None


def examples() -> dict[str, dict]:
    client = TestClient(create_app())
    personas = {p["user_id"]: p for p in client.get("/api/personas").json()}

    def ask(profile: dict, user_id: str | None = None, mode: str = "ours") -> dict:
        body = {"profile": profile, "mode": mode, "limit": 10}
        if user_id:
            body["user_id"] = user_id
        response = client.post("/api/recommendations", json=body)
        response.raise_for_status()
        return response.json()

    services = {"tasks": ["marketing"], "industries": ["services"]}
    seller = personas["persona-seller"]
    outside = {"free_text": "Открыть автосервис в Казахстане"}
    return {
        "health": client.get("/api/health").json(),
        "services": ask(services),
        "services_baseline": ask(services, mode="baseline"),
        "seller": ask(seller.get("profile", {}), seller["user_id"]),
        "seller_persona": seller,
        "outside": ask(outside),
        "cold": ask({}),
    }


# Screenshot and PDF ---------------------------------------------------------------------------


def find_browser(explicit: str | None) -> str | None:
    for candidate in [explicit, *BROWSERS]:
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def screenshot(browser: str, target: Path) -> bool:
    port = free_port()
    env = {**os.environ, "EVENTS_PATH": str(DIST / "screenshot-events.jsonl")}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "alfarec.api.main:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            return False
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--window-size=1440,900", "--virtual-time-budget=8000",
             "--blink-settings=preferredColorScheme=1",
             f"--screenshot={target}", f"http://127.0.0.1:{port}/"],
            check=True, timeout=120, capture_output=True,
        )  # fmt: skip
        return target.is_file()
    finally:
        server.terminate()
        server.wait(timeout=30)


def print_pdf(browser: str, source: Path, target: Path) -> None:
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=4000", f"--print-to-pdf={target}", source.as_uri()],
        check=True, timeout=180, capture_output=True,
    )  # fmt: skip


# Rendering ------------------------------------------------------------------------------------


def card(entry: dict, number: int | None = None) -> str:
    item = entry["item"]
    reasons = [r["text"] for r in entry.get("reasons", []) if r.get("text")][:2]
    meta = [FORMAT_LABELS.get(item["format"], item["format"])]
    if item.get("duration_min"):
        meta.append(f"{item['duration_min']} мин")
    meta.append(ACCESS_LABELS.get(item["access"], item["access"]))
    step = entry.get("next_step") or {}
    badge = f'<span class="num">{number}</span>' if number else ""
    why = "".join(f"<li>{e(text)}</li>" for text in reasons)
    return (
        f'<div class="card">{badge}<div class="meta">{e(" · ".join(meta))}</div>'
        f'<div class="title">{e(item["title"])}</div><ul class="why">{why}</ul>'
        f'<div class="step">{e(step.get("label", ""))} →</div></div>'
    )


def ru(text: str) -> str:
    """Russian decimal comma: 0.499 -> 0,499; identifiers such as e5-small stay as they are."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", text)


def verdict(rows: list[list[str]]) -> str:
    """One-line takeaway of the held-out table: our nDCG@10 against the baseline's."""
    if not rows or "nDCG@10" not in rows[0]:
        return ""
    col = rows[0].index("nDCG@10")
    values = {}
    for row in rows[1:]:
        key = "base" if "Бейзлайн" in row[0] else "ours" if row[0].startswith("Наша") else ""
        try:
            values.setdefault(key, float(row[col]))
        except (ValueError, IndexError):
            continue
    ours, base = values.get("ours"), values.get("base")
    if ours is None or not base:
        return ""
    return ru(
        f"Ручная релевантность на отложенных профилях: nDCG@10 {ours:.3f} против {base:.3f}"
        f" у бейзлайна — в {ours / base:.1f} раза выше"
    )


def table(rows: list[list[str]], highlight: int | None = None) -> str:
    head, *body = rows
    out = ["<table><thead><tr>", *(f"<th>{e(c)}</th>" for c in head), "</tr></thead><tbody>"]
    for index, row in enumerate(body):
        cls = ' class="hl"' if index == highlight else ""
        out += [f"<tr{cls}>", *(f"<td>{e(ru(c))}</td>" for c in row), "</tr>"]
    return "".join(out + ["</tbody></table>"])


def slide(title: str, body: str, kicker: str = "", cls: str = "") -> str:
    kick = f'<div class="kicker">{e(kicker)}</div>' if kicker else ""
    return f'<section class="slide {cls}">{kick}<h2>{e(title)}</h2>{body}</section>'


def bullets(items: list[str]) -> str:
    return "<ul class='bullets'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def render(data: dict, tables: dict, ml: dict, cases: str, has_shot: bool) -> str:
    models = ml.get("Протокол", [])
    labels = ml.get("Метки и контроль", [])
    manual_count, manual = holdout_table(tables)
    final_report = ROOT / "reports" / "b06_final.json"
    final = json.loads(final_report.read_text("utf-8")) if final_report.is_file() else {}
    initial = final.get("metrics", {}).get("holdout", {}).get("initial", {}).get("ndcg10")
    weight_note = (
        ru(f"Начальные веса дали nDCG@10 {initial:.3f}; настройка немного уступила им. ")
        if isinstance(initial, (int, float))
        else ""
    )
    model_id = packaged_model()
    commit = commit_label()
    next_course = tables.get("Следующий курс по поведению", [])
    editorial = tables.get("Редакторские подборки", [])
    profiles = tables.get("Списки по 36 тестовым профилям (без разметки)", [])
    services, base = data["services"], data["services_baseline"]
    seller, outside, cold = data["seller"], data["outside"], data["cold"]
    route = services.get("route", [])
    health = data["health"]
    note = services.get("route_note") or ""
    slides: list[str] = []

    def accuracy(field: str) -> str:
        """Accepted accuracy of a label field from the B04 label check table."""
        for row in labels[1:]:
            if row[0].startswith(field):
                return row[2].split("/")[0].strip()
        return "нет данных"

    slides.append(
        '<section class="slide title-slide">'
        '<div class="kicker">Альфа-Хакатон · трек 1 · персональные рекомендации</div>'
        "<h1>Маршрут вместо ленты</h1>"
        "<p class='lead'>Предприниматель за минуту и без анкеты получает короткий маршрут из"
        " материалов «Альфа-Курса» под свою задачу, стадию и опыт — с объяснением, почему"
        " каждый материал подходит и что сделать дальше.</p>"
        f"<div class='repo'>Репозиторий: <a href='{REPO_URL}'>{REPO_URL}</a>"
        + (f" · коммит <code>{e(commit)}</code>" if commit else "")
        + "</div>"
        "<ol class='check'><li><code>docker compose up --build</code> — ключи, дамп и GPU не"
        " нужны</li><li>Открыть <code>http://localhost:8000</code> — стартовый набор сразу,"
        " API — <code>/docs</code></li><li>Пройти 7 шагов раздела «Как проверить» в"
        " README</li></ol></section>"
    )
    slides.append(
        slide(
            "Проблема и пользователь",
            "<div class='cols'><div>"
            + bullets(
                [
                    "1497 открытых материалов семи типов: статьи, шаблоны документов, курсы,"
                    " вебинары, подкасты, глоссарий, калькуляторы.",
                    "Пользователь — предприниматель: от «думаю открыть» до «расширяюсь»,"
                    " ИП, ООО, самозанятые.",
                    "Одному нужен короткий ответ, другому шаблон, третьему курс. Лента по теме"
                    " этого не различает.",
                ]
            )
            + "</div><div class='quote'>«Что мне прочитать, скачать и пройти, чтобы решить"
            " мою задачу — и в каком порядке?»</div></div>",
            "Зачем",
        )
    )
    slides.append(
        slide(
            "Что увидели в данных",
            bullets(
                [
                    "Дамп PostgreSQL на 62 таблицы читаем напрямую из Python, без СУБД.",
                    "Стадия бизнеса, уровень и детальная задача почти не размечены: три широких"
                    " тега покрывают ~97% материалов. Их восстанавливаем сами.",
                    "Поведения мало: только записи на курсы (10 050), уроки и оценки; у 79%"
                    " пользователей ровно один курс, просмотров статей нет.",
                    "Есть независимая разметка платформы: 3996 редакторских связей «смотрите"
                    " также» — используем для честной проверки, а не как вход.",
                    "Платные уроки лежат в дампе целиком — в артефакты они не попадают.",
                ]
            ),
            "Данные",
        )
    )
    route_cards = "".join(card(entry, n) for n, entry in enumerate(route, 1))
    slides.append(
        slide(
            "Решение: маршрут «понять → сделать → углубиться»",
            "<p class='sub'>Профиль: задача «Продвижение и маркетинг», сфера «Услуги». Ответ"
            " API на этом коммите:</p>"
            f"<div class='route'>{route_cards}</div>"
            f"<p class='note'>{e(note)}</p>",
            "Продукт",
        )
    )
    slides.append(
        slide(
            "Архитектура",
            "<div class='arch'>"
            "<div class='box'><b>Офлайн, один раз</b><br>дамп → каталог без текстов уроков;"
            " на GPU — векторы материалов, метки и граф соседей; резервный индекс TF-IDF,"
            " переходы между курсами, персоны, оценка</div>"
            "<div class='arrow'>→</div>"
            "<div class='box'><b>artifacts/ в git</b><br>catalog.parquet · ml/ (векторы, метки,"
            " граф) · retrieval_tfidf · course_transitions.json · personas.json</div>"
            "<div class='arrow'>→</div>"
            "<div class='box'><b>Docker, CPU</b><br>FastAPI: рекомендации, события, мета;"
            " страница без сборки фронтенда; веса модели скачиваются при сборке образа</div>"
            "</div>"
            + bullets(
                [
                    "Одна команда запуска, один сервис, порт 8000; СУБД, GPU и внешние API не"
                    " нужны, после сборки образа не нужна и сеть.",
                    "Без весов модели сервис переходит на TF-IDF, без индекса — на бейзлайн;"
                    " фактический режим виден в <code>/api/health</code>.",
                    "Примеры в этом PDF получены в режиме"
                    f" <code>recommender: {e(health.get('recommender', '?'))}</code>,"
                    f" <code>encoder: {e(health.get('encoder', '?'))}</code>,"
                    f" материалов: {e(str(health.get('catalog_items', '?')))}.",
                ]
            ),
            "Как устроено",
        )
    )
    slides.append(
        slide(
            "ML: разметка и представления материалов",
            "<p class='sub'>Эмбеддинги: поиск редакторских связей платформы по названию и"
            " описанию. Модель выбрана только по tune и пределам 500 МБ и 150 мс; holdout"
            f" приведён описательно. В продукте — <b>{e(model_id or 'TF-IDF')}</b>.</p>"
            + (
                table(columns(models, [0, 2, 3, 4, 5, 6]), row_index(models, "e5-small"))
                if models
                else "<p>нет данных</p>"
            )
            + "<p class='sub'>Стадия, уровень и задачи: 120 материалов размечены вручную"
            " вслепую, в ячейках accuracy / macro-F1</p>"
            + (table(labels) if labels else "<p>нет данных</p>")
            + "<p class='todo'>Превосходство нейросети над TF-IDF не заявляется: на tune TF-IDF"
            " чуть сильнее, выборка мала. 120 меток использованы для выбора источников меток —"
            " это калибровка, а не независимая проверка. Подробности — reports/b04_ml.md.</p>",
            "ML",
        )
    )
    slides.append(
        slide(
            "Ранжирование и объяснения",
            "<div class='cols'><div>"
            + bullets(
                [
                    "Кандидаты: близкие по запросу, материалы задач профиля, соседи истории,"
                    " курсы.",
                    "Линейный скоринг из 12 интерпретируемых признаков: задача, близость, стадия,"
                    " уровень, сфера, формат, история, переходы, качество, популярность,"
                    " свежесть, уместность платного.",
                    "Переранжирование: MMR, в первых пяти не больше одного платного и двух одной"
                    " темы, не меньше двух форматов.",
                    "Каждый вклад признака → код причины → фраза «Почему вам».",
                ]
            )
            + "</div><div>"
            + (card(services["more"][0]) if services.get("more") else "")
            + "</div></div>",
            "Алгоритм",
        )
    )
    cold_cards = "".join(card(entry) for entry in cold.get("more", [])[:3])
    clar = outside.get("clarification") or {}
    options = "".join(f"<span class='chip'>{e(o)}</span>" for o in clar.get("options", []))
    slides.append(
        slide(
            "Холодный старт и мало данных",
            "<div class='cols'><div><p class='sub'>Без профиля — стартовый набор и «План на"
            f" неделю» ({len(cold.get('route', []))} коротких материалов)</p>"
            f"<div class='stack compact'>{cold_cards}</div></div>"
            "<div><p class='sub'>«Открыть автосервис в Казахстане» — честное уточнение вместо"
            f" выдуманного совпадения</p><div class='clar'><b>{e(clar.get('question', ''))}</b>"
            f"<div>{options}</div></div></div></div>",
            "Без сведений",
        )
    )
    seller_entries = seller.get("route", []) or seller.get("more", [])
    seller_cards = "".join(card(entry, n) for n, entry in enumerate(seller_entries[:3], 1))
    persona = data["seller_persona"]
    shot = "<img class='shot' src='page.png' alt='Страница сервиса'>" if has_shot else ""
    slides.append(
        slide(
            "Примеры работы",
            "<div class='cols'><div>"
            f"{shot}<p class='sub'>Первый вход: стартовый набор и карточка «Подберём точнее за 3"
            " вопроса»</p></div><div>"
            f"<p class='sub'>Персона «{e(persona['title'])}». {e(persona.get('legend', ''))}"
            " Её маршрут:</p>"
            f"<div class='stack compact'>{seller_cards}</div></div></div>",
            "Демонстрация",
        )
    )
    base_titles = "".join(f"<li>{e(x['item']['title'])}</li>" for x in base.get("more", [])[:5])
    slides.append(
        slide(
            "Метрики против бейзлайна",
            (f"<p class='verdict'>{e(verdict(manual))}</p>" if verdict(manual) else "")
            + "<div class='cols'><div>"
            f"<p class='sub'>Следующий курс, отложенных пользователей: {e(cases)}; переходы"
            " построены по другим</p>"
            + (table(next_course, highlight=0) if next_course else "<p>нет данных</p>")
            + "<p class='sub'>Редакторские подборки платформы, диагностика</p>"
            + (table(editorial, highlight=0) if editorial else "")
            + "</div><div><p class='sub'>36 тестовых профилей, без ручной разметки</p>"
            + (table(profiles[:5]) if profiles else "")
            + (
                f"<p class='sub'>Ручная релевантность: holdout, {e(manual_count)} профилей</p>"
                + table(columns(manual, [0, 1, 2, 3]), highlight=0)
                if manual
                else "<p class='sub'>Бейзлайн для «Продвижение · Услуги»:</p>"
                f"<ul class='plain'>{base_titles}</ul>"
            )
            + "</div></div>"
            + (
                "<p class='todo'>Веса выбраны только на tune. "
                + e(weight_note)
                + "Выборка мала, есть явные unjudgeable. Редакторские связи уже использовались"
                " при выборе модели; бизнес-эффект не измерен.</p>"
                if manual
                else "<p class='todo'>Ручная релевантность не измерена: nDCG@10, Precision@5 и"
                " MRR появятся после оценок двух участников. Таблицы на слайде описывают"
                " поведение и форму выдачи, а не её релевантность.</p>"
            ),
            "Оценка · reports/eval.md",
        )
    )
    slides.append(
        slide(
            "План A/B-проверки и внедрения",
            bullets(
                [
                    "Гипотеза: маршрут с объяснениями повышает долю новых покупок курса или"
                    " «Курс+» за 28 дней; вторая цель — активные дни.",
                    "Контроль — подборка по теме того же каталога; назначение — по стабильному"
                    " хешу пользователя, 50/50, анализ intention-to-treat.",
                    "Пример объёма: при конверсии 1% и росте до 1,2% нужно 42 693 пользователя"
                    " на группу.",
                    "Порядок: офлайн-оценка → A/A и журналы → пилот 10% → A/B → раскатка с"
                    " ограничителями (жалобы, отписки, скорость).",
                ]
            ),
            "docs/ab-test.md",
        )
    )
    slides.append(
        slide(
            "Ограничения и развитие",
            "<div class='cols'><div><b>Честно о текущей версии</b>"
            + bullets(
                (
                    [
                        "Отложенная выборка мала, профили использовались ранее."
                        " Есть явные unjudgeable.",
                    ]
                    if manual
                    else [
                        "Ручная релевантность выдачи не измерена: форма выдачи и поведенческая"
                        " проверка её не подтверждают.",
                        "Веса ранжирования заданы вручную, подобран только вес переходов между"
                        " курсами.",
                    ]
                )
                + [
                    "Вклад отдельных признаков не измерен: абляций нет.",
                    "Популярность взята из полного снимка — офлайн-оценка не временной backtest.",
                    f"Метки неточны: на ручной проверке стадии — {e(accuracy('Стадии'))},"
                    f" задачи — {e(accuracy('Задачи'))} accuracy.",
                    "Профиля пользователя в данных нет: персоны и тестовые профили — наши.",
                ]
            )
            + "</div><div><b>Дальше</b>"
            + bullets(
                [
                    (
                        "Новая независимая оценочная выборка и learning-to-rank"
                        " на событиях после запуска."
                        if manual
                        else "Подбор весов по ручным оценкам, затем learning-to-rank на событиях."
                    ),
                    "Исторические снимки признаков для честной временной оценки.",
                    "Онлайн-обновление профиля по «Уже знаю» и «Не то».",
                    "Интеграция с платформой и A/B-тест.",
                ]
            )
            + "</div></div>",
            "Итог",
        )
    )
    css = (Path(__file__).parent / "slides.css").read_text(encoding="utf-8")
    return (
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        f"<title>Маршрут вместо ленты</title><style>{css}</style></head><body>"
        + "".join(slides)
        + "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browser")
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args(argv)
    DIST.mkdir(exist_ok=True)
    report = ROOT / "reports" / "eval.md"
    ml_report = ROOT / "reports" / "b04_ml.md"
    browser = find_browser(args.browser)
    shot = DIST / "page.png"
    has_shot = bool(browser) and not args.skip_pdf and screenshot(browser, shot)
    page = DIST / "presentation.html"
    data = examples()
    encoder = data["health"].get("encoder")
    if packaged_model() and encoder != "dense":
        print(
            f"Warning: examples use encoder '{encoder}', not the packaged model;"
            " install it before building the submission PDF (see README)",
            file=sys.stderr,
        )
    page.write_text(
        render(data, markdown_tables(report), markdown_tables(ml_report), case_count(report),
               has_shot),
        encoding="utf-8",
    )  # fmt: skip
    print(f"Slides: {page}")
    if args.skip_pdf:
        return 0
    if not browser:
        print("Edge or Chrome not found; pass --browser", file=sys.stderr)
        return 1
    pdf = DIST / "alfa-course-recommendations.pdf"
    print_pdf(browser, page, pdf)
    print(f"PDF: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
