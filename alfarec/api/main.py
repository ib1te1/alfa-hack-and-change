import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from alfarec.api.events import EventStore
from alfarec.api.schemas import (
    EventAccepted,
    HealthResponse,
    MetaOption,
    MetaResponse,
    Persona,
    RecommendationRequest,
)
from alfarec.recsys.baseline import TopicBaseline
from alfarec.recsys.catalog import DEFAULT_CATALOG_PATH, load_catalog, with_base_url
from alfarec.recsys.contracts import Event, Item, RecommendationResponse, Recommender
from alfarec.recsys.explain import explain_response
from alfarec.recsys.ranking import DEFAULT_INDEX_PATH, load_ranker
from alfarec.recsys.taxonomy import Taxonomy, load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
DEFAULT_EVENTS_PATH = ROOT / "data" / "events" / "events.jsonl"
DEFAULT_PERSONAS_PATH = ROOT / "artifacts" / "personas.json"
TIME_BUDGETS = [
    MetaOption(id="quick", label="5 минут"),
    MetaOption(id="hour", label="Час"),
    MetaOption(id="course", label="Готов к курсу"),
]

logger = logging.getLogger(__name__)


@dataclass
class Services:
    taxonomy: Taxonomy
    catalog: dict[str, Item]
    recommender: Recommender | None
    events: EventStore
    personas: list[Persona]
    # "ranking" when the B05 ranker runs; "baseline" when its index could not be loaded.
    recommender_kind: Literal["ranking", "baseline", "unavailable"] = "unavailable"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def load_personas(path: Path) -> list[Persona]:
    if not path.is_file():
        return []
    personas = [Persona.model_validate(entry) for entry in json.loads(path.read_text("utf-8"))]
    for persona in personas:
        if any(event.user_id != persona.user_id for event in persona.history):
            raise ValueError(f"Persona {persona.id} has events of another user")
    return personas


def build_services(
    catalog_path: Path | None = None,
    events_path: Path | None = None,
    personas_path: Path | None = None,
    index_path: Path | None = None,
) -> Services:
    taxonomy = load_taxonomy()
    catalog_path = catalog_path or _env_path("CATALOG_PATH", DEFAULT_CATALOG_PATH)
    items: list[Item] = []
    base_url = os.environ.get("PLATFORM_BASE_URL")
    recommender: Recommender | None = None
    kind: Literal["ranking", "baseline", "unavailable"] = "unavailable"
    if catalog_path.is_file():
        items = load_catalog(catalog_path)
        if base_url:
            items = with_base_url(items, base_url)
        try:
            recommender = load_ranker(
                catalog_path, index_path or _env_path("INDEX_PATH", DEFAULT_INDEX_PATH),
                base_url=base_url,
            )  # fmt: skip
            kind = "ranking"
        except (OSError, ValueError) as error:
            # A stale or missing index must not take the page down: serve the baseline and
            # report it in /api/health.
            logger.error("Ranking index unavailable, serving the baseline: %s", error)
            recommender, kind = TopicBaseline(items, taxonomy), "baseline"
    else:
        logger.warning("Catalog artifact is missing: %s", catalog_path)
    personas = load_personas(personas_path or _env_path("PERSONAS_PATH", DEFAULT_PERSONAS_PATH))
    seed = [event for persona in personas for event in persona.history]
    events = EventStore(
        events_path or _env_path("EVENTS_PATH", DEFAULT_EVENTS_PATH),
        seed,
        memory_only=[persona.user_id for persona in personas],
    )
    return Services(
        taxonomy=taxonomy,
        catalog={item.item_id: item for item in items},
        recommender=recommender,
        events=events,
        personas=personas,
        recommender_kind=kind,
    )


def _options(entries: dict) -> list[MetaOption]:
    return [
        MetaOption(id=key, label=entry.label, description=entry.description)
        for key, entry in entries.items()
    ]


def create_app(services: Services | None = None) -> FastAPI:
    state = services or build_services()
    application = FastAPI(
        title="Альфа-Курс: рекомендации",
        version="0.2.0",
        description=(
            "Подбор материалов «Альфа-Курса» под задачу, стадию и опыт предпринимателя. "
            "До подключения ML-ранжирования режим ours использует промежуточную выдачу "
            "на основе бейзлайна по темам."
        ),
    )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса"})

    @application.middleware("http")
    async def revalidate_page(request: Request, call_next):
        response = await call_next(request)
        # The page and its scripts change between releases; ETag keeps revalidation cheap.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ranking = state.recommender_kind == "ranking"
        return HealthResponse(
            status="ok" if ranking else "degraded",
            ready=state.recommender is not None,
            artifacts_loaded=bool(state.catalog),
            catalog_items=len(state.catalog),
            encoder="tfidf" if ranking else "unavailable",
            recommender=state.recommender_kind,
        )

    @application.get("/api/meta", response_model=MetaResponse)
    def meta() -> MetaResponse:
        taxonomy = state.taxonomy
        return MetaResponse(
            stages=_options(taxonomy.stages),
            levels=_options(taxonomy.levels),
            tasks=_options(taxonomy.tasks),
            industries=_options(taxonomy.industries),
            formats=_options(taxonomy.formats),
            time_budgets=TIME_BUDGETS,
        )

    @application.get("/api/personas", response_model=list[Persona])
    def personas() -> list[Persona]:
        return state.personas

    @application.post("/api/recommendations", response_model=RecommendationResponse)
    def recommendations(request: RecommendationRequest) -> RecommendationResponse:
        if state.recommender is None:
            raise HTTPException(503, "Каталог не загружен: соберите artifacts/catalog.parquet")
        unknown = [
            value
            for value, known in (
                *((task, state.taxonomy.tasks) for task in request.profile.tasks),
                *((industry, state.taxonomy.industries) for industry in request.profile.industries),
            )
            if value not in known
        ]
        if unknown:
            raise HTTPException(422, f"Неизвестные значения профиля: {', '.join(unknown)}")
        history = state.events.history(request.user_id)
        response = state.recommender.recommend(
            request.profile, history, request.mode, request.limit
        )
        return explain_response(response, state.taxonomy)

    @application.post("/api/events", response_model=EventAccepted, status_code=201)
    def events(event: Event) -> EventAccepted:
        if event.item_id not in state.catalog:
            raise HTTPException(404, f"Материал {event.item_id} не найден")
        state.events.add(event)
        return EventAccepted(events_for_user=len(state.events.history(event.user_id)))

    @application.get("/api/items/{item_id}", response_model=Item)
    def item(item_id: str) -> Item:
        found = state.catalog.get(item_id)
        if found is None:
            raise HTTPException(404, f"Материал {item_id} не найден")
        return found

    @application.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")

    application.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return application


app = create_app()
