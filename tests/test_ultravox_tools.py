"""Tests for the Ultravox tool webhook endpoints."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


SECRET = "test-secret-xyz"


@pytest.fixture(autouse=True)
def _set_tool_secret(monkeypatch):
    monkeypatch.setenv("ULTRAVOX_TOOL_SECRET", SECRET)


@pytest.fixture
def client():
    from routers.ultravox_tools import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _sign_headers(call_id: str, body: bytes) -> dict[str, str]:
    ts = str(int(time.time()))
    msg = f"{ts}.{call_id}.".encode("utf-8") + body
    sig = hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return {
        "X-Ultravox-Call-ID": call_id,
        "X-Ultravox-Signature-Timestamp": ts,
        "X-Ultravox-Signature": sig,
        "Content-Type": "application/json",
    }


class TestSignatureGating:
    def test_missing_signature_rejects_with_401(self, client):
        resp = client.post("/api/ultravox/tools/datetime", json={})
        assert resp.status_code == 401

    def test_bad_signature_rejects_with_401(self, client):
        body = b"{}"
        headers = _sign_headers("call-1", body)
        headers["X-Ultravox-Signature"] = "deadbeef"
        resp = client.post("/api/ultravox/tools/datetime", headers=headers, content=body)
        assert resp.status_code == 401

    def test_missing_tool_secret_returns_503(self, monkeypatch, client):
        # Remove secret AFTER the client fixture has built the app, so the
        # endpoint sees an empty env when it runs.
        monkeypatch.delenv("ULTRAVOX_TOOL_SECRET", raising=False)
        # We don't even need valid signature headers — the secret check
        # comes first.
        resp = client.post("/api/ultravox/tools/datetime", content=b"{}")
        assert resp.status_code == 503

    def test_invalid_json_body_returns_400(self, client):
        body = b"{not-valid-json}"
        resp = client.post(
            "/api/ultravox/tools/datetime",
            headers=_sign_headers("call-bad-json", body),
            content=body,
        )
        assert resp.status_code == 400


class TestDatetime:
    def test_valid_signature_returns_uae_datetime(self, client):
        body = b"{}"
        resp = client.post(
            "/api/ultravox/tools/datetime",
            headers=_sign_headers("call-dt-1", body),
            content=body,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "date" in data
        assert "time" in data
        assert data["timezone"].startswith("Gulf")


class TestSearchCourses:
    def test_returns_courses_for_beauty(self, client):
        body = json.dumps({"query": "beauty"}).encode("utf-8")
        resp = client.post(
            "/api/ultravox/tools/search-courses",
            headers=_sign_headers("call-sc-1", body),
            content=body,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["count"] >= 1


class TestSearchFaq:
    def test_returns_faq_for_khda(self, client):
        body = json.dumps({"query": "KHDA"}).encode("utf-8")
        resp = client.post(
            "/api/ultravox/tools/search-faq",
            headers=_sign_headers("call-faq-1", body),
            content=body,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestCaptureLead:
    def test_missing_name_returns_error_payload(self, client):
        body = json.dumps({"phone": "0561234567"}).encode("utf-8")
        resp = client.post(
            "/api/ultravox/tools/capture-lead",
            headers=_sign_headers("call-lead-1", body),
            content=body,
        )
        assert resp.status_code == 200  # tool returns 200 with error status in payload
        data = resp.json()
        assert data["status"] == "error"

    def test_full_payload_captures_lead(self, client):
        body = json.dumps({
            "name": "Aisha Khan",
            "phone": "0561234567",
            "email": "aisha@example.com",
            "course_interest": "CIDESCO Beauty Therapy",
            "notes": "",
        }).encode("utf-8")
        resp = client.post(
            "/api/ultravox/tools/capture-lead",
            headers=_sign_headers("call-lead-2", body),
            content=body,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "captured"
        assert "Aisha" in data["message"]

    def test_lead_capture_updates_session_state(self, client):
        from core.conversation_state import default_store, Stage

        call_id = "call-lead-state-1"
        body = json.dumps({"name": "Test User", "phone": "0561234567"}).encode("utf-8")
        resp = client.post(
            "/api/ultravox/tools/capture-lead",
            headers=_sign_headers(call_id, body),
            content=body,
        )
        assert resp.status_code == 200
        state = default_store().peek(call_id)
        assert state is not None
        assert state.phone_captured is True
        assert state.stage == Stage.IDENTIFIED
