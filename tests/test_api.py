from fastapi.testclient import TestClient

from alfarec.api.main import create_app


def test_health_distinguishes_running_server_from_ready_recommender() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "scaffold",
            "ready": False,
            "artifacts_loaded": False,
            "encoder": "unavailable",
            "recommender": "stub",
        }


def test_documented_api_does_not_advertise_unimplemented_recommendations() -> None:
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
        assert "/api/health" in schema["paths"]
        assert "/api/recommendations" not in schema["paths"]
        assert client.post("/api/recommendations", json={}).status_code == 404
        assert client.get("/docs").status_code == 200
