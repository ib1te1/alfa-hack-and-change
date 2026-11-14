from typing import Literal

from fastapi import FastAPI

from alfarec.recsys.contracts import Contract


class HealthResponse(Contract):
    status: Literal["scaffold"] = "scaffold"
    ready: bool = False
    artifacts_loaded: bool = False
    encoder: Literal["unavailable"] = "unavailable"
    recommender: Literal["stub"] = "stub"


def create_app() -> FastAPI:
    application = FastAPI(
        title="Альфа-Курс: рекомендации",
        version="0.1.0",
        description="Каркас сервиса. Рекомендательная система ещё не подключена.",
    )

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    return application


app = create_app()
