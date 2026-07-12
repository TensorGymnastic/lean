from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_echo_returns_payload() -> None:
    client = TestClient(app)
    response = client.post("/v1/chat/echo", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json() == {"response": "hello"}
