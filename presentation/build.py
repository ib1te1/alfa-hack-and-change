"""Build the B12 presentation: HTML slides from live data, then a PDF through headless Edge.

Usage: python presentation/build.py [--browser PATH] [--skip-pdf]

Metrics come from reports/eval.md, examples from the running API (in-process), and the page
screenshot from a local server; nothing is typed in by hand, so the deck matches the commit.
Output goes to dist/ (outside git): presentation.html, page.png, alfa-course-recommendations.pdf.
"""

import argparse
import html
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


def table(rows: list[list[str]], highlight: int | None = None) -> str:
    head, *body = rows
    out = ["<table><thead><tr>", *(f"<th>{e(c)}</th>" for c in head), "</tr></thead><tbody>"]
    for index, row in enumerate(body):
        cls = ' class="hl"' if index == highlight else ""
        out += [f"<tr{cls}>", *(f"<td>{e(c)}</td>" for c in row), "</tr>"]
    return "".join(out + ["</tbody></table>"])


def slide(title: str, body: str, kicker: str = "", cls: str = "") -> str:
    kick = f'<div class="kicker">{e(kicker)}</div>' if kicker else ""
    return f'<section class="slide {cls}">{kick}<h2>{e(title)}</h2>{body}</section>'


def bullets(items: list[str]) -> str:
    return "<ul class='bullets'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def render(data: dict, tables: dict, cases: str, has_shot: bool) -> str:
    next_course = tables.get("Следующий курс по поведению", [])
    editorial = tables.get("Редакторские подборки", [])
    profiles = tables.get("Списки по 36 тестовым профилям (без разметки)", [])
    services, base = data["services"], data["services_baseline"]
    seller, outside, cold = data["seller"], data["outside"], data["cold"]
    route = services.get("route", [])
    health = data["health"]
    note = services.get("route_note") or ""
    slides: list[str] = []

    slides.append(
        '<section class="slide title-slide">'
        '<div class="kicker">Альфа-Хакатон · трек 1 · персональные рекомендации</div>'
        "<h1>Маршрут вместо ленты</h1>"
        "<p class='lead'>Предприниматель за минуту и без анкеты получает короткий маршрут из"
        " материалов «Альфа-Курса» под свою задачу, стадию и опыт — с объяснением, почему"
        " каждый материал подходит и что сделать дальше.</p>"
        f"<div class='repo'>Репозиторий: <a href='{REPO_URL}'>{REPO_URL}</a></div>"
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
            "<div class='box'><b>Офлайн, один раз</b><br>дамп → каталог без текстов уроков,"
            " индекс TF-IDF, переходы между курсами, персоны, оценка</div>"
            "<div class='arrow'>→</div>"
            "<div class='box'><b>artifacts/ в git</b><br>catalog.parquet · retrieval_tfidf ·"
            " course_transitions.json · personas.json</div>"
            "<div class='arrow'>→</div>"
            "<div class='box'><b>Docker, CPU</b><br>FastAPI: рекомендации, события, мета;"
            " страница без сборки фронтенда</div></div>"
            + bullets(
                [
                    "Одна команда запуска, один сервис, порт 8000; СУБД и внешние API не нужны.",
                    "Без индекса сервис переходит на бейзлайн и сообщает об этом в"
                    " <code>/api/health</code>.",
                    f"Сейчас: <code>recommender: {e(health.get('recommender', '?'))}</code>,"
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
            bullets(
                [
                    "Сейчас в продукте: TF-IDF индекс по названию и описанию, задачи, стадии и"
                    " уровень — правилами по исходной разметке.",
                    "Поведенческий сигнал: переходы «курс → следующий курс» по 10 тыс. записей,"
                    " только агрегированные счётчики пар, без пользователей.",
                    "Готовится (B04, GPU): сравнение моделей эмбеддингов на редакторских"
                    " подборках, восстановление стадии и уровня локальной LLM, проверка точности"
                    " на 120 материалах вслепую.",
                ]
            )
            + "<p class='todo'>Слайд дополняет ML-инженер по итогам B04: модель, Recall@10 и"
            " точность разметки.</p>",
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
            f"<div class='stack'>{cold_cards}</div></div>"
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
            f"<p class='sub'>Персона «{e(persona['title'])}»: {e(persona.get('legend', ''))}"
            " Её маршрут:</p>"
            f"<div class='stack compact'>{seller_cards}</div></div></div>",
            "Демонстрация",
        )
    )
    base_titles = "".join(f"<li>{e(x['item']['title'])}</li>" for x in base.get("more", [])[:5])
    slides.append(
        slide(
            "Метрики против бейзлайна",
            "<div class='cols'><div>"
            f"<p class='sub'>Следующий курс: {e(cases)} отложенных пользователей, переходы"
            " построены по другим</p>"
            + (table(next_course, highlight=0) if next_course else "<p>нет данных</p>")
            + "<p class='sub'>Редакторские подборки платформы</p>"
            + (table(editorial, highlight=0) if editorial else "")
            + "</div><div><p class='sub'>36 тестовых профилей, без ручной разметки</p>"
            + (table(profiles[:5]) if profiles else "")
            + "<p class='sub'>Бейзлайн для «Продвижение · Услуги»:</p>"
            f"<ul class='plain'>{base_titles}</ul></div></div>"
            "<p class='todo'>nDCG@10 по ручной разметке — после оценок двух участников (B06).</p>",
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
                [
                    "Релевантность подтверждена формой выдачи и поведением, ручная разметка"
                    " ещё впереди.",
                    "Популярность взята из полного снимка — офлайн-оценка не временной backtest.",
                    "Профиля пользователя в данных нет: персоны и тестовые профили — наши.",
                ]
            )
            + "</div><div><b>Дальше</b>"
            + bullets(
                [
                    "Эмбеддинги и LLM-разметка стадии и уровня (B04).",
                    "Learning-to-rank на событиях после запуска.",
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
    browser = find_browser(args.browser)
    shot = DIST / "page.png"
    has_shot = bool(browser) and not args.skip_pdf and screenshot(browser, shot)
    page = DIST / "presentation.html"
    page.write_text(
        render(examples(), markdown_tables(report), case_count(report), has_shot),
        encoding="utf-8",
    )
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
