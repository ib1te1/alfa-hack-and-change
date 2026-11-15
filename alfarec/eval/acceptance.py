"""Acceptance run of the core scenarios A02-A10 (spec section 14) on the current commit.

Usage: python -m alfarec.eval.acceptance [--output reports/acceptance.md]

Each scenario calls the API in process and states what was checked. A01 (clean start on a
second machine with Docker) cannot be checked here and is reported as not run. The report
names the commit, so B13 can attach it to the submitted version.
"""

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from alfarec.api.main import create_app

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "acceptance.md"
REQUIRED_FILES = [
    "README.md",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    "requirements.txt",
    ".env.example",
]
LEAK_TESTS = [
    "tests/test_explain.py::test_paid_cards_show_only_public_fields",
    "tests/test_explain.py::test_api_never_returns_lesson_text_for_paid_materials",
]


@dataclass
class Result:
    scenario: str
    passed: bool | None
    evidence: str


class Checks:
    def __init__(self) -> None:
        self.client = TestClient(create_app())
        self.personas = {p["user_id"]: p for p in self.client.get("/api/personas").json()}

    def ask(self, profile: dict, user_id: str | None = None, mode: str = "ours") -> dict:
        body: dict = {"profile": profile, "mode": mode, "limit": 10}
        if user_id:
            body["user_id"] = user_id
        response = self.client.post("/api/recommendations", json=body)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def cards(answer: dict) -> list[dict]:
        return answer.get("route", []) + answer.get("more", []) + answer.get("quick", [])

    def a02(self) -> Result:
        answer = self.ask({"tasks": ["marketing"]})
        route = answer["route"]
        formats = {c["item"]["format"] for c in route}
        on_task = all(any(r["code"] == "task_match" for r in c["reasons"]) for c in route)
        passed = (
            3 <= len(route) <= 5 and len(formats) >= 2 and on_task and bool(answer["route_note"])
        )
        return Result(
            "A02. Новый пользователь с задачей",
            passed,
            f"«Продвижение»: маршрут {len(route)} шагов, форматов {len(formats)},"
            f" у всех шагов task_match: {on_task}; порядок: «{answer['route_note']}»",
        )

    def a03(self) -> Result:
        persona = self.personas["persona-seller"]
        with_history = self.ask(persona.get("profile", {}), persona["user_id"])
        without = self.ask(persona.get("profile", {}))
        done = {e["item_id"] for e in persona.get("history", []) if e["type"] == "complete"}
        shown = {c["item"]["item_id"] for c in self.cards(with_history)}
        continues = any(
            r["code"] == "continues_history" for c in self.cards(with_history) for r in c["reasons"]
        )
        differs = [c["item"]["item_id"] for c in self.cards(with_history)] != [
            c["item"]["item_id"] for c in self.cards(without)
        ]
        passed = not (done & shown) and continues and differs
        return Result(
            "A03. История",
            passed,
            f"«{persona['title']}»: пройденных в выдаче {len(done & shown)} из {len(done)},"
            f" причина «продолжает…»: {continues}, отличается от выдачи без истории:"
            f" {differs}",
        )

    def a04(self) -> Result:
        answer = self.ask({})
        formats = {c["item"]["format"] for c in answer["more"]}
        passed = len(answer["route"]) == 5 and len(formats) >= 3
        return Result(
            "A04. Холодный старт",
            passed,
            f"Пустой профиль: план на неделю {len(answer['route'])} материалов,"
            f" стартовый набор {len(answer['more'])} карточек {len(formats)} форматов",
        )

    def a05(self) -> Result:
        answer = self.ask({"free_text": "Открыть автосервис в Казахстане"})
        clarification = answer.get("clarification") or {}
        options = clarification.get("options", [])
        passed = 2 <= len(options) <= 4 and bool(answer["more"]) and not answer["route"]
        return Result(
            "A05. Мало данных / нет совпадения",
            passed,
            f"«Открыть автосервис в Казахстане»: вопрос «{clarification.get('question')}»,"
            f" вариантов {len(options)}, общая подборка {len(answer['more'])},"
            " маршрута нет",
        )

    def a06(self) -> Result:
        profiles = [{"tasks": [t]} for t in ("marketing", "taxes_reporting", "team_hiring")]
        profiles += [{}, {"free_text": "как выбрать налоговый режим для ИП"}]
        cards = [c for p in profiles for c in self.cards(self.ask(p))]
        # Spec section 14: every card carries two or three reasons with text.
        bad_reasons = [
            c for c in cards if not 2 <= len([r for r in c["reasons"] if r.get("text")]) <= 3
        ]
        bad_steps = [
            c
            for c in cards
            if not str((c.get("next_step") or {}).get("url", "")).startswith("https://")
        ]
        passed = not bad_reasons and not bad_steps
        examples = "; ".join(sorted({c["item"]["title"] for c in bad_reasons})[:3])
        return Result(
            "A06. Объяснения и шаг",
            passed,
            f"{len(cards)} карточек пяти профилей: без 2–3 причин с текстом"
            f" {len(bad_reasons)}{f' (например: {examples})' if examples else ''},"
            f" без ссылки шага {len(bad_steps)}",
        )

    def a07(self) -> Result:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *LEAK_TESTS],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        last = (run.stdout.strip().splitlines() or ["нет вывода"])[-1]
        return Result(
            "A07. Платное содержание",
            run.returncode == 0,
            f"Автотесты публичных полей и отсутствия текстов уроков: {last}",
        )

    def a08(self) -> Result:
        profile = {"tasks": ["marketing"], "industries": ["services"]}
        ours = [c["item"]["item_id"] for c in self.cards(self.ask(profile))]
        base = [c["item"]["item_id"] for c in self.cards(self.ask(profile, mode="baseline"))]
        # "Reproduced by a command": run it into a scratch file and compare the tables.
        with tempfile.TemporaryDirectory() as scratch:
            fresh = Path(scratch) / "eval.md"
            run = subprocess.run(
                [sys.executable, "-m", "alfarec.eval", "--output", str(fresh)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            rebuilt = fresh.read_text(encoding="utf-8") if fresh.is_file() else ""
        committed = (ROOT / "reports" / "eval.md").read_text(encoding="utf-8")
        same = stable_lines(rebuilt) == stable_lines(committed) and bool(rebuilt)
        passed = run.returncode == 0 and same and bool(base) and ours != base
        return Result(
            "A08. Сравнение с бейзлайном",
            passed,
            f"`python -m alfarec.eval` завершился с кодом {run.returncode}, отчёт совпадает с"
            f" закоммиченным (кроме задержки): {same}; режим baseline отвечает"
            f" ({len(base)} карточек) и отличается от нашего",
        )

    def a09(self) -> Result:
        steps = [
            {},
            {"stage": "launch"},
            {"stage": "launch", "industries": ["services"]},
            {"stage": "launch", "industries": ["services"], "tasks": ["marketing"]},
            {
                "stage": "launch",
                "industries": ["services"],
                "tasks": ["marketing"],
                "experience": "beginner",
                "time_budget": "quick",
            },
        ]
        lists = [[c["item"]["item_id"] for c in self.cards(self.ask(p))] for p in steps]
        changed = sum(a != b for a, b in zip(lists, lists[1:], strict=False))
        passed = all(lists) and changed == len(steps) - 1
        return Result(
            "A09. Онбординг необязателен",
            passed,
            f"Пустой профиль и четыре шага онбординга по одному ответу: все отвечают,"
            f" выдача меняется после {changed} из {len(steps) - 1} ответов (API;"
            " обновление страницы проверяется в браузере)",
        )

    def a10(self) -> Result:
        missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
        # The organisers' dump and anything under data/; tests/fixtures holds a synthetic
        # two-row dump for the reader tests, which is not case data.
        paths = ["data", "*alfa_course_hackathon*"]
        listing = subprocess.run(
            ["git", "ls-files", *paths], cwd=ROOT, capture_output=True, text=True
        )
        log = subprocess.run(
            ["git", "log", "--all", "--format=%h", "--", *paths],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if listing.returncode or log.returncode:
            # Without a repository (a Docker image) an empty listing proves nothing.
            return Result("A10. Сдаваемый комплект", None, "Нет репозитория git: не проверялось")
        tracked, history = listing.stdout.split(), log.stdout.split()
        passed = not missing and not tracked and not history
        return Result(
            "A10. Сдаваемый комплект",
            passed,
            f"Файлов не хватает: {missing or 'нет'}; дамп в дереве: {len(tracked)},"
            f" в истории: {len(history)}. PDF и доступ жюри проверяются вручную",
        )


def stable_lines(report: str) -> list[str]:
    """Report lines without the latency row, which changes with the machine."""
    return [line for line in report.splitlines() if "Задержка" not in line]


def commit() -> str:
    run = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return run.stdout.strip() + (" (есть незакоммиченные изменения)" if dirty else "")


def render(results: list[Result], sha: str) -> str:
    mark = {True: "пройден", False: "**не пройден**", None: "не проверялся"}
    lines = [
        "# Приёмка обязательного ядра",
        "",
        f"Коммит `{sha}`. Создано `python -m alfarec.eval.acceptance`: сценарии раздела 14 ТЗ"
        " проверяются вызовами API в процессе. Это не заменяет ручной прогон в браузере и чистый"
        " запуск на второй машине.",
        "",
        "| Сценарий | Итог | Что проверено |",
        "|---|---|---|",
    ]
    lines += [f"| {r.scenario} | {mark[r.passed]} | {r.evidence} |" for r in results]
    return "\n".join(lines) + "\n"


def run_all() -> list[Result]:
    checks = Checks()
    results = [
        Result(
            "A01. Чистый запуск",
            None,
            "Нужны Docker и вторая машина со свежим клоном; здесь не запускался",
        )
    ]
    scenarios: list[Callable[[], Result]] = [
        checks.a02,
        checks.a03,
        checks.a04,
        checks.a05,
        checks.a06,
        checks.a07,
        checks.a08,
        checks.a09,
        checks.a10,
    ]
    for scenario in scenarios:
        try:
            results.append(scenario())
        except Exception as error:  # noqa: BLE001 - one failed scenario must not hide the rest
            name = scenario.__name__.upper()
            results.append(Result(name, False, f"Ошибка: {type(error).__name__}: {error}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    results = run_all()
    args.output.write_text(render(results, commit()), encoding="utf-8", newline="\n")
    for result in results:
        print(f"{result.scenario}: {result.passed}")
    return 0 if all(r.passed is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
