"""Tests for the public /api/public/ultravox-call endpoint."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _ultravox_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
    monkeypatch.setenv("ULTRAVOX_API_KEY", "ux-test-key")
    monkeypatch.setenv("ULTRAVOX_AGENT_ID", "ux-agent-id")
    monkeypatch.setenv("ULTRAVOX_TOOL_SECRET", "ux-tool-secret")


@pytest.fixture
def client(monkeypatch):
    # Patch httpx so we don't actually hit Ultravox
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        return httpx.Response(
            201,
            json={
                "callId": "call-test-1",
                "joinUrl": "https://join.ultravox.ai/call-test-1?token=t",
                "created": "2026-05-20T10:00:00Z",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    # Import after env is set
    from main import app
    return TestClient(app)


class TestUltravoxCallEndpoint:
    def test_returns_join_url_for_english(self, client):
        resp = client.get("/api/public/ultravox-call?lang=en")
        assert resp.status_code == 200
        data = resp.json()
        assert data["joinUrl"].startswith("https://")
        assert data["callId"] == "call-test-1"
        assert data["lang"] == "en"

    def test_returns_join_url_for_arabic(self, client):
        resp = client.get("/api/public/ultravox-call?lang=ar")
        assert resp.status_code == 200
        data = resp.json()
        assert data["lang"] == "ar"

    def test_unknown_lang_defaults_to_english(self, client):
        resp = client.get("/api/public/ultravox-call?lang=zz")
        assert resp.status_code == 200
        assert resp.json()["lang"] == "en"

    def test_source_is_passed_through(self, monkeypatch):
        # Use our own client + capturing fake_post for this test, so we can
        # inspect the body the gateway sent to Ultravox.
        captured = {}

        async def fake_post(self, url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return httpx.Response(
                201,
                json={"callId": "c", "joinUrl": "https://x", "created": "t"},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        from main import app
        client = TestClient(app)

        resp = client.get("/api/public/ultravox-call?lang=en&source=q2-campaign")
        assert resp.status_code == 200
        assert captured["json"]["metadata"]["source"] == "q2-campaign"

    def test_missing_config_returns_503(self, monkeypatch):
        # Even though the autouse fixture set ULTRAVOX_API_KEY, remove it
        # for this test so the endpoint's config-missing guard fires.
        monkeypatch.delenv("ULTRAVOX_API_KEY", raising=False)
        from main import app
        client = TestClient(app)
        resp = client.get("/api/public/ultravox-call?lang=en")
        assert resp.status_code == 503

    def test_ultravox_failure_returns_502(self, monkeypatch):
        async def fake_post(self, url, headers=None, json=None, timeout=None):
            return httpx.Response(
                401,
                json={"error": "unauthorized"},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        from main import app
        client = TestClient(app)
        resp = client.get("/api/public/ultravox-call?lang=en")
        assert resp.status_code == 502
