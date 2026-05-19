"""Tests for the Ultravox REST API wrapper."""
from __future__ import annotations

import json

import httpx
import pytest

from core.ultravox_client import UltravoxClient, UltravoxConfig


@pytest.fixture
def config():
    return UltravoxConfig(
        api_key="ultravox-test-key",
        agent_id="agent-uuid-123",
        base_url="https://api.ultravox.ai",
    )


@pytest.mark.asyncio
async def test_create_call_posts_to_agent_calls_endpoint(config, monkeypatch):
    captured = {}

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            201,
            json={
                "callId": "call-xyz-789",
                "joinUrl": "https://join.ultravox.ai/call-xyz-789?token=abc",
                "created": "2026-05-20T10:00:00Z",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = UltravoxClient(config)
    result = await client.create_call(lang="en", source="web")

    assert captured["url"].endswith("/api/agents/agent-uuid-123/calls")
    assert captured["headers"]["X-API-Key"] == "ultravox-test-key"
    assert captured["json"]["languageHint"] == "en-US"
    assert captured["json"]["medium"]["webRtc"]["dataMessages"]["transcript"] is True
    assert result["callId"] == "call-xyz-789"
    assert result["joinUrl"].startswith("https://")


@pytest.mark.asyncio
async def test_create_call_arabic_sets_voice_override(config, monkeypatch):
    captured = {}

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return httpx.Response(
            201,
            json={"callId": "c", "joinUrl": "https://x", "created": "t"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = UltravoxClient(config)
    await client.create_call(lang="ar", source="web")

    body = captured["json"]
    assert body["languageHint"] == "ar"
    assert "voiceOverrides" in body
    assert "inworld" in body["voiceOverrides"]
    assert body["voiceOverrides"]["inworld"]["voiceId"]  # non-empty


@pytest.mark.asyncio
async def test_create_call_includes_source_in_metadata(config, monkeypatch):
    captured = {}

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return httpx.Response(
            201,
            json={"callId": "c", "joinUrl": "https://x", "created": "t"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = UltravoxClient(config)
    await client.create_call(lang="en", source="campaign-q2")

    assert captured["json"]["metadata"]["source"] == "campaign-q2"


@pytest.mark.asyncio
async def test_create_call_raises_on_non_2xx(config, monkeypatch):
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        return httpx.Response(
            401,
            json={"error": "unauthorized"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = UltravoxClient(config)
    with pytest.raises(httpx.HTTPStatusError):
        await client.create_call(lang="en", source="web")
