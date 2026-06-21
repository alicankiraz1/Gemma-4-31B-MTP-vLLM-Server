from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from prometheus_fastapi_instrumentator import Instrumentator


def main() -> None:
    app = FastAPI()
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Exercise FastAPI's included-router route objects, which is where the
    # vLLM HTTP stack previously tripped instrumentator route-name matching.
    app.include_router(router)
    Instrumentator().instrument(app).expose(app)
    response = TestClient(app).get("/health")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


if __name__ == "__main__":
    main()
