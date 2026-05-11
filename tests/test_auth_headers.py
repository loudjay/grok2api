from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.auth import verify_api_key
from app.services.api_keys import api_key_manager


def _make_app():
    app = FastAPI()

    @app.get("/protected")
    async def protected(_token: str | None = Depends(verify_api_key)):
        return {"ok": True}

    return app


def test_verify_api_key_accepts_x_api_key_header(monkeypatch):
    async def _fake_init():
        return None

    monkeypatch.setattr(api_key_manager, "init", _fake_init)
    monkeypatch.setattr(api_key_manager, "validate_key", lambda token: None)
    monkeypatch.setattr("app.core.auth.get_config", lambda key, default=None: "global-key" if key == "app.api_key" else default)

    client = TestClient(_make_app())
    resp = client.get("/protected", headers={"x-api-key": "global-key"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_verify_api_key_accepts_managed_api_key(monkeypatch):
    async def _fake_init():
        return None

    monkeypatch.setattr(api_key_manager, "init", _fake_init)
    monkeypatch.setattr(
        api_key_manager,
        "validate_key",
        lambda token: {"key": token, "is_active": True} if token == "managed-key" else None,
    )
    monkeypatch.setattr("app.core.auth.get_config", lambda key, default=None: "" if key == "app.api_key" else default)

    client = TestClient(_make_app())
    resp = client.get("/protected", headers={"Authorization": "Bearer managed-key"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

