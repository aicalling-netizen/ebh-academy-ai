# Shakira Ultravox Realtime — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel realtime voice stack for the Shakira EBH Academy agent using Ultravox's native API, leaving the current LiveKit+Claude stack untouched.

**Architecture:** Ultravox cloud handles speech-to-speech (STT+LLM+TTS in one connection); webhook-based HTTP tools POST to the existing FastAPI gateway on EC2 and reuse the existing `tools/*.py` handlers; a new web client opens WebRTC directly to Ultravox.

**Tech Stack:** Python 3.11+, FastAPI 0.133, httpx 0.28 (already pinned), Ultravox REST API + JS client SDK, HMAC-SHA256 signature validation, pytest 9.0 + pytest-asyncio (auto mode).

**Spec:** [docs/superpowers/specs/2026-05-20-shakira-ultravox-realtime-design.md](../specs/2026-05-20-shakira-ultravox-realtime-design.md)

**Target repo:** `E:\AI websocket\ebh-academy-ai\` (separate git repo, branch `master`). This worktree (`quirky-hypatia-844cc1`) holds only the spec and plan docs.

---

## Pre-flight checklist

- [ ] **Step 0a:** Confirm you are working in the `ebh-academy-ai` repo, not the PAM worktree. Run `git remote -v` from `E:\AI websocket\ebh-academy-ai\` — should show the academy repo.
- [ ] **Step 0b:** Create a feature branch off `master`:

```bash
cd "E:\AI websocket\ebh-academy-ai"
git checkout -b feat/ultravox-realtime
```

- [ ] **Step 0c:** Verify dependencies present (Ultravox needs only `httpx` which is already pinned in `requirements.txt`):

```bash
grep "^httpx" requirements.txt
# Expected: httpx==0.28.1
```

- [ ] **Step 0d:** Run the existing test suite to establish a green baseline:

```bash
pytest --basetemp=.pytest_tmp
# Expected: all tests pass
```

If baseline is red, stop and surface to user before proceeding.

---

## File Map

**Create (in `ebh-academy-ai/`):**

```
routers/
  __init__.py                        # package marker
  ultravox_tools.py                  # 4 webhook endpoints + signature middleware
core/
  ultravox_client.py                 # Ultravox REST API wrapper (create_call)
  ultravox_signature.py              # HMAC-SHA256 signature verification
  conversation_state.py              # In-memory per-call session state
data/
  system_prompt_ultravox.txt         # Layer 1 persona core (~80 lines)
demo/
  shakira_realtime.html              # New web client using Ultravox JS SDK
tests/
  test_ultravox_signature.py         # HMAC verify edge cases
  test_ultravox_tools.py             # 4 endpoints, signature gating
  test_ultravox_client.py            # Call creation request shape
  test_conversation_state.py         # Session state transitions
  test_ultravox_call_endpoint.py     # /api/public/ultravox-call integration
```

**Modify (in `ebh-academy-ai/`):**

```
main.py                              # Mount routers, add /api/public/ultravox-call
.env.example                         # Add ULTRAVOX_* env vars
```

**Untouched (current Shakira keeps running):**

```
agent.py
tools/*.py
core/db.py, core/rag_pipeline.py, core/time_context.py
demo/index.html
```

---

## Task 1: HMAC signature helper

**Why first:** Every webhook endpoint depends on this. Pure function, easy to TDD.

**Files:**
- Create: `core/ultravox_signature.py`
- Test: `tests/test_ultravox_signature.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_ultravox_signature.py`:

```python
"""Tests for Ultravox webhook signature verification."""
from __future__ import annotations

import hashlib
import hmac

import pytest

from core.ultravox_signature import verify_signature


SECRET = "test-secret-abc123"


def _sign(timestamp: str, call_id: str, body: bytes, secret: str = SECRET) -> str:
    msg = f"{timestamp}.{call_id}.".encode("utf-8") + body
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


class TestVerifySignature:
    def test_valid_signature_passes(self):
        body = b'{"query":"beauty"}'
        ts = "1747756800"
        call_id = "call-abc-123"
        sig = _sign(ts, call_id, body)
        assert verify_signature(body=body, timestamp=ts, call_id=call_id, signature=sig, secret=SECRET) is True

    def test_wrong_signature_fails(self):
        body = b'{"query":"beauty"}'
        ts = "1747756800"
        call_id = "call-abc-123"
        assert verify_signature(body=body, timestamp=ts, call_id=call_id, signature="deadbeef", secret=SECRET) is False

    def test_tampered_body_fails(self):
        ts = "1747756800"
        call_id = "call-abc-123"
        good_sig = _sign(ts, call_id, b'{"query":"beauty"}')
        tampered = b'{"query":"hacked"}'
        assert verify_signature(body=tampered, timestamp=ts, call_id=call_id, signature=good_sig, secret=SECRET) is False

    def test_wrong_secret_fails(self):
        body = b'{"query":"beauty"}'
        ts = "1747756800"
        call_id = "call-abc-123"
        sig = _sign(ts, call_id, body, secret="other-secret")
        assert verify_signature(body=body, timestamp=ts, call_id=call_id, signature=sig, secret=SECRET) is False

    def test_constant_time_comparison(self):
        body = b'x'
        ts = "1"
        call_id = "c"
        sig = _sign(ts, call_id, body)
        # If a length-different forged signature crashes, we used == instead of hmac.compare_digest
        assert verify_signature(body=body, timestamp=ts, call_id=call_id, signature="ab", secret=SECRET) is False
```

- [ ] **Step 1.2: Run test, confirm it fails**

```bash
pytest tests/test_ultravox_signature.py -v --basetemp=.pytest_tmp
# Expected: FAIL — ModuleNotFoundError: No module named 'core.ultravox_signature'
```

- [ ] **Step 1.3: Implement minimal code**

Create `core/ultravox_signature.py`:

```python
"""HMAC-SHA256 signature verification for Ultravox webhook callbacks.

Ultravox signs tool webhooks per its sharedSecrets mechanism:
  X-Ultravox-Call-ID:            <call uuid>
  X-Ultravox-Signature-Timestamp: <unix seconds>
  X-Ultravox-Signature:           <hex sha256 hmac>

The signed payload is `timestamp.call_id.<body>`.
"""
from __future__ import annotations

import hashlib
import hmac


def verify_signature(
    *,
    body: bytes,
    timestamp: str,
    call_id: str,
    signature: str,
    secret: str,
) -> bool:
    """Return True iff the given signature matches the body under `secret`.

    Uses constant-time comparison to avoid timing attacks.
    """
    if not secret or not signature or not timestamp or not call_id:
        return False
    msg = f"{timestamp}.{call_id}.".encode("utf-8") + body
    expected = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

- [ ] **Step 1.4: Run tests, confirm they pass**

```bash
pytest tests/test_ultravox_signature.py -v --basetemp=.pytest_tmp
# Expected: 5 passed
```

- [ ] **Step 1.5: Commit**

```bash
git add core/ultravox_signature.py tests/test_ultravox_signature.py
git commit -m "feat(ultravox): add HMAC signature verification helper"
```

---

## Task 2: Conversation state tracker

**Why:** Spec §5 Layer 2 moves Shakira's state machine (turn count, identified, qualified) out of the LLM context and into app code.

**Files:**
- Create: `core/conversation_state.py`
- Test: `tests/test_conversation_state.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_conversation_state.py`:

```python
"""Tests for per-call conversation state tracking."""
from __future__ import annotations

import pytest

from core.conversation_state import (
    SessionState,
    SessionStore,
    Stage,
)


class TestSessionState:
    def test_initial_state(self):
        s = SessionState(call_id="abc")
        assert s.call_id == "abc"
        assert s.stage == Stage.ANONYMOUS
        assert s.turn_count == 0
        assert s.phone_captured is False
        assert s.care_mode is False

    def test_record_turn_increments_count(self):
        s = SessionState(call_id="abc")
        s.record_turn()
        s.record_turn()
        assert s.turn_count == 2

    def test_mark_lead_captured_transitions_to_identified(self):
        s = SessionState(call_id="abc")
        s.mark_lead_captured()
        assert s.stage == Stage.IDENTIFIED
        assert s.phone_captured is True

    def test_care_mode_overrides_other_state(self):
        s = SessionState(call_id="abc")
        s.mark_lead_captured()
        s.enable_care_mode()
        assert s.care_mode is True
        assert s.stage == Stage.IDENTIFIED  # underlying stage preserved


class TestSessionStore:
    def test_get_creates_on_first_access(self):
        store = SessionStore()
        s = store.get("call-1")
        assert s.call_id == "call-1"
        assert s.turn_count == 0

    def test_get_returns_same_instance(self):
        store = SessionStore()
        a = store.get("call-1")
        b = store.get("call-1")
        assert a is b

    def test_different_calls_are_isolated(self):
        store = SessionStore()
        a = store.get("call-1")
        b = store.get("call-2")
        a.record_turn()
        assert b.turn_count == 0

    def test_drop_removes_session(self):
        store = SessionStore()
        store.get("call-1").record_turn()
        store.drop("call-1")
        assert store.get("call-1").turn_count == 0  # fresh state
```

- [ ] **Step 2.2: Run test, confirm fail**

```bash
pytest tests/test_conversation_state.py -v --basetemp=.pytest_tmp
# Expected: FAIL — ModuleNotFoundError
```

- [ ] **Step 2.3: Implement minimal code**

Create `core/conversation_state.py`:

```python
"""Per-call session state for the Ultravox realtime Shakira agent.

The current LiveKit Shakira tracks conversation state inside the LLM prompt
(ANONYMOUS → IDENTIFIED → QUALIFIED → BOOKED, plus CARE override). Ultravox's
smaller model handles a layered short prompt better; we keep state here in
app code and inject hints into the agent via Ultravox templateContext when a
state change is relevant.

This module is intentionally in-memory and single-process. If the gateway
ever runs multi-instance, swap SessionStore for a Redis-backed implementation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Stage(str, Enum):
    ANONYMOUS = "anonymous"
    IDENTIFIED = "identified"
    QUALIFIED = "qualified"
    BOOKED = "booked"


@dataclass
class SessionState:
    """Mutable state for a single Ultravox call."""

    call_id: str
    stage: Stage = Stage.ANONYMOUS
    turn_count: int = 0
    phone_captured: bool = False
    care_mode: bool = False
    lang: str = "en"
    metadata: dict = field(default_factory=dict)

    def record_turn(self) -> None:
        self.turn_count += 1

    def mark_lead_captured(self) -> None:
        self.phone_captured = True
        if self.stage == Stage.ANONYMOUS:
            self.stage = Stage.IDENTIFIED

    def mark_qualified(self) -> None:
        if self.stage in (Stage.ANONYMOUS, Stage.IDENTIFIED):
            self.stage = Stage.QUALIFIED

    def mark_booked(self) -> None:
        self.stage = Stage.BOOKED

    def enable_care_mode(self) -> None:
        self.care_mode = True


class SessionStore:
    """Thread-safe in-memory store of SessionState by call_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, call_id: str) -> SessionState:
        with self._lock:
            existing = self._sessions.get(call_id)
            if existing is not None:
                return existing
            state = SessionState(call_id=call_id)
            self._sessions[call_id] = state
            return state

    def drop(self, call_id: str) -> None:
        with self._lock:
            self._sessions.pop(call_id, None)

    def peek(self, call_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(call_id)


# Module-level singleton used by the gateway. Tests construct their own
# SessionStore for isolation.
_default_store = SessionStore()


def default_store() -> SessionStore:
    return _default_store
```

- [ ] **Step 2.4: Run tests, confirm pass**

```bash
pytest tests/test_conversation_state.py -v --basetemp=.pytest_tmp
# Expected: 8 passed
```

- [ ] **Step 2.5: Commit**

```bash
git add core/conversation_state.py tests/test_conversation_state.py
git commit -m "feat(ultravox): add in-memory conversation state tracker"
```

---

## Task 3: Ultravox REST API client

**Why:** The gateway needs to create a call on Ultravox to get the `joinUrl` the web client uses for WebRTC.

**Files:**
- Create: `core/ultravox_client.py`
- Test: `tests/test_ultravox_client.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_ultravox_client.py`:

```python
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

    def fake_post(self, url, headers=None, json=None, timeout=None):
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

    def fake_post(self, url, headers=None, json=None, timeout=None):
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

    def fake_post(self, url, headers=None, json=None, timeout=None):
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
    def fake_post(self, url, headers=None, json=None, timeout=None):
        return httpx.Response(
            401,
            json={"error": "unauthorized"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = UltravoxClient(config)
    with pytest.raises(httpx.HTTPStatusError):
        await client.create_call(lang="en", source="web")
```

- [ ] **Step 3.2: Run test, confirm fail**

```bash
pytest tests/test_ultravox_client.py -v --basetemp=.pytest_tmp
# Expected: FAIL — ModuleNotFoundError
```

- [ ] **Step 3.3: Implement minimal code**

Create `core/ultravox_client.py`:

```python
"""Thin async wrapper around the Ultravox REST API.

We only use the create-call endpoint:
  POST /api/agents/{agent_id}/calls

Auth: X-API-Key header. See https://docs.ultravox.ai/.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("academy.ultravox.client")

_DEFAULT_INWORLD_AR_VOICE = "WcxyRPjVQcpVYmceBQO4Helb"  # Aisha — reused from current Faseeh setup


@dataclass
class UltravoxConfig:
    api_key: str
    agent_id: str
    base_url: str = "https://api.ultravox.ai"
    max_duration: str = "1800s"
    tool_secret: str = ""  # passed to Ultravox so signed callbacks can be verified

    @classmethod
    def from_env(cls) -> "UltravoxConfig":
        return cls(
            api_key=os.getenv("ULTRAVOX_API_KEY", "").strip(),
            agent_id=os.getenv("ULTRAVOX_AGENT_ID", "").strip(),
            base_url=os.getenv("ULTRAVOX_BASE_URL", "https://api.ultravox.ai").strip(),
            max_duration=os.getenv("ULTRAVOX_MAX_DURATION", "1800s").strip(),
            tool_secret=os.getenv("ULTRAVOX_TOOL_SECRET", "").strip(),
        )


class UltravoxClient:
    """Async client for creating Ultravox agent calls."""

    def __init__(self, config: UltravoxConfig) -> None:
        if not config.api_key:
            raise ValueError("UltravoxConfig.api_key is required")
        if not config.agent_id:
            raise ValueError("UltravoxConfig.agent_id is required")
        self._config = config

    async def create_call(
        self,
        *,
        lang: str,
        source: str,
    ) -> dict:
        """Create an Ultravox call and return the response JSON.

        The returned dict contains at least: callId, joinUrl, created.
        Raises httpx.HTTPStatusError on non-2xx responses.
        """
        url = f"{self._config.base_url}/api/agents/{self._config.agent_id}/calls"
        body = self._build_request_body(lang=lang, source=source)
        headers = {"X-API-Key": self._config.api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=body, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Ultravox call created: callId=%s lang=%s source=%s",
            data.get("callId"),
            lang,
            source,
        )
        return data

    def _build_request_body(self, *, lang: str, source: str) -> dict:
        lang_hint = "ar" if lang == "ar" else "en-US"
        body: dict = {
            "languageHint": lang_hint,
            "initialOutputMedium": "MESSAGE_MEDIUM_VOICE",
            "maxDuration": self._config.max_duration,
            "firstSpeakerSettings": {
                "agent": {
                    "text": (
                        "مرحباً، أنا شاكيرا من أكاديمية إي بي إتش. كيف يمكنني مساعدتك؟"
                        if lang == "ar"
                        else "Hi, this is Shakira from EBH Academy — how can I help?"
                    ),
                    "uninterruptible": False,
                }
            },
            "medium": {
                "webRtc": {
                    "dataMessages": {
                        "transcript": True,
                        "state": True,
                        "clientToolInvocation": True,
                    }
                }
            },
            "metadata": {"source": source},
        }
        if self._config.tool_secret:
            body["sharedSecrets"] = [self._config.tool_secret]
        if lang == "ar":
            voice_id = os.getenv("ULTRAVOX_AR_INWORLD_VOICE_ID", _DEFAULT_INWORLD_AR_VOICE)
            body["voiceOverrides"] = {"inworld": {"voiceId": voice_id}}
        return body
```

- [ ] **Step 3.4: Run tests, confirm pass**

```bash
pytest tests/test_ultravox_client.py -v --basetemp=.pytest_tmp
# Expected: 4 passed
```

- [ ] **Step 3.5: Commit**

```bash
git add core/ultravox_client.py tests/test_ultravox_client.py
git commit -m "feat(ultravox): add REST API client for creating agent calls"
```

---

## Task 4: Webhook tool endpoints router

**Why:** Ultravox POSTs to these endpoints when it decides to call a tool. Each must verify the HMAC signature and delegate to the existing tool handler.

**Files:**
- Create: `routers/__init__.py`
- Create: `routers/ultravox_tools.py`
- Test: `tests/test_ultravox_tools.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_ultravox_tools.py`:

```python
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
```

- [ ] **Step 4.2: Run test, confirm fail**

```bash
pytest tests/test_ultravox_tools.py -v --basetemp=.pytest_tmp
# Expected: FAIL — ModuleNotFoundError: routers.ultravox_tools
```

- [ ] **Step 4.3: Implement minimal code**

Create `routers/__init__.py`:

```python
"""HTTP routers for the EBH Academy gateway."""
```

Create `routers/ultravox_tools.py`:

```python
"""Webhook endpoints for Ultravox agent tool calls.

Each endpoint:
  1. Verifies the HMAC-SHA256 signature using ULTRAVOX_TOOL_SECRET.
  2. Delegates to the existing tool handler in `tools/*.py`.
  3. Returns the handler's JSON dict, which Ultravox speaks back to the user.

Endpoints intentionally use POST + raw body so signature verification operates
on the exact bytes Ultravox signed (Starlette's request.json() reparses).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from core.conversation_state import default_store
from core.ultravox_signature import verify_signature

logger = logging.getLogger("academy.ultravox.tools")

router = APIRouter(prefix="/api/ultravox/tools", tags=["ultravox"])


async def _verify_request(
    request: Request,
    call_id: str | None,
    timestamp: str | None,
    signature: str | None,
) -> tuple[bytes, str]:
    """Read body and verify the Ultravox signature. Return (body_bytes, call_id)."""
    secret = os.getenv("ULTRAVOX_TOOL_SECRET", "").strip()
    if not secret:
        logger.error("ULTRAVOX_TOOL_SECRET not configured — refusing webhook")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="server not configured")
    body = await request.body()
    if not call_id or not timestamp or not signature:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing signature headers")
    if not verify_signature(
        body=body,
        timestamp=timestamp,
        call_id=call_id,
        signature=signature,
        secret=secret,
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")
    return body, call_id


def _parse_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}")
    return parsed if isinstance(parsed, dict) else {}


@router.post("/datetime")
async def tool_datetime(
    request: Request,
    x_ultravox_call_id: str | None = Header(default=None, alias="X-Ultravox-Call-ID"),
    x_ultravox_signature_timestamp: str | None = Header(default=None, alias="X-Ultravox-Signature-Timestamp"),
    x_ultravox_signature: str | None = Header(default=None, alias="X-Ultravox-Signature"),
):
    body, call_id = await _verify_request(
        request, x_ultravox_call_id, x_ultravox_signature_timestamp, x_ultravox_signature,
    )
    args = _parse_json(body)
    default_store().get(call_id).record_turn()
    from tools.current_datetime import _handle
    return await _handle(args)


@router.post("/search-courses")
async def tool_search_courses(
    request: Request,
    x_ultravox_call_id: str | None = Header(default=None, alias="X-Ultravox-Call-ID"),
    x_ultravox_signature_timestamp: str | None = Header(default=None, alias="X-Ultravox-Signature-Timestamp"),
    x_ultravox_signature: str | None = Header(default=None, alias="X-Ultravox-Signature"),
):
    body, call_id = await _verify_request(
        request, x_ultravox_call_id, x_ultravox_signature_timestamp, x_ultravox_signature,
    )
    args = _parse_json(body)
    default_store().get(call_id).record_turn()
    from tools.course_inquiry import _handle
    return await _handle(args)


@router.post("/search-faq")
async def tool_search_faq(
    request: Request,
    x_ultravox_call_id: str | None = Header(default=None, alias="X-Ultravox-Call-ID"),
    x_ultravox_signature_timestamp: str | None = Header(default=None, alias="X-Ultravox-Signature-Timestamp"),
    x_ultravox_signature: str | None = Header(default=None, alias="X-Ultravox-Signature"),
):
    body, call_id = await _verify_request(
        request, x_ultravox_call_id, x_ultravox_signature_timestamp, x_ultravox_signature,
    )
    args = _parse_json(body)
    default_store().get(call_id).record_turn()
    from tools.academy_faq import _handle
    return await _handle(args)


@router.post("/capture-lead")
async def tool_capture_lead(
    request: Request,
    x_ultravox_call_id: str | None = Header(default=None, alias="X-Ultravox-Call-ID"),
    x_ultravox_signature_timestamp: str | None = Header(default=None, alias="X-Ultravox-Signature-Timestamp"),
    x_ultravox_signature: str | None = Header(default=None, alias="X-Ultravox-Signature"),
):
    body, call_id = await _verify_request(
        request, x_ultravox_call_id, x_ultravox_signature_timestamp, x_ultravox_signature,
    )
    args = _parse_json(body)
    state = default_store().get(call_id)
    state.record_turn()
    from tools.enrollment_lead import _handle
    result = await _handle(args)
    if result.get("status") == "captured":
        state.mark_lead_captured()
    return result
```

- [ ] **Step 4.4: Run tests, confirm pass**

```bash
pytest tests/test_ultravox_tools.py -v --basetemp=.pytest_tmp
# Expected: 7 passed
```

- [ ] **Step 4.5: Commit**

```bash
git add routers/__init__.py routers/ultravox_tools.py tests/test_ultravox_tools.py
git commit -m "feat(ultravox): add webhook tool endpoints (4 tools + HMAC gating)"
```

---

## Task 5: Public call-creation endpoint on the gateway

**Why:** The web client needs an unauthenticated endpoint that returns an Ultravox `joinUrl`. Reuses the existing rate-limiter.

**Files:**
- Modify: `main.py` (mount router, add endpoint)
- Test: `tests/test_ultravox_call_endpoint.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_ultravox_call_endpoint.py`:

```python
"""Tests for the public /api/public/ultravox-call endpoint."""
from __future__ import annotations

import os

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
    def fake_post(self, url, headers=None, json=None, timeout=None):
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

    def test_source_is_passed_through(self, client):
        resp = client.get("/api/public/ultravox-call?lang=en&source=q2-campaign")
        assert resp.status_code == 200
```

- [ ] **Step 5.2: Run test, confirm fail**

```bash
pytest tests/test_ultravox_call_endpoint.py -v --basetemp=.pytest_tmp
# Expected: FAIL — endpoint not yet defined (404 on the route)
```

- [ ] **Step 5.3: Modify `main.py`**

Add the router import + mount, and the new endpoint. Place the new code just before the `# ── Public LiveKit token endpoint ──` section header.

Find this exact line in `main.py`:

```python
# ── Public LiveKit token endpoint ────────────────────────────────────────
```

Insert the following block *immediately above* that line:

```python
# ── Ultravox realtime stack ──────────────────────────────────────────────

from routers.ultravox_tools import router as _ultravox_tools_router
app.include_router(_ultravox_tools_router)


@app.get("/talk-realtime", response_class=HTMLResponse)
async def serve_talk_realtime():
    """Serve the Ultravox realtime web call page."""
    page = Path(__file__).parent / "demo" / "shakira_realtime.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Realtime talk page not found")


@app.get("/api/public/ultravox-call")
async def api_public_ultravox_call(
    request: Request,
    _rate_limit: None = Depends(_check_public_rate),
):
    """Create an Ultravox call and return the joinUrl for the web client."""
    lang = (request.query_params.get("lang") or "en").strip().lower()
    if lang not in {"en", "ar"}:
        lang = "en"
    source = (request.query_params.get("source") or "web").strip() or "web"

    try:
        from core.ultravox_client import UltravoxClient, UltravoxConfig

        config = UltravoxConfig.from_env()
        if not config.api_key or not config.agent_id:
            raise HTTPException(status_code=503, detail="Ultravox not configured")

        client = UltravoxClient(config)
        result = await client.create_call(lang=lang, source=source)
        return {
            "joinUrl": result.get("joinUrl"),
            "callId": result.get("callId"),
            "agentId": config.agent_id,
            "lang": lang,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ultravox call creation failed")
        raise HTTPException(status_code=502, detail="Failed to create realtime call")
```

- [ ] **Step 5.4: Run tests, confirm pass**

```bash
pytest tests/test_ultravox_call_endpoint.py -v --basetemp=.pytest_tmp
# Expected: 4 passed
```

- [ ] **Step 5.5: Run full suite to confirm no regressions**

```bash
pytest --basetemp=.pytest_tmp
# Expected: all tests pass (existing + new)
```

- [ ] **Step 5.6: Commit**

```bash
git add main.py tests/test_ultravox_call_endpoint.py
git commit -m "feat(ultravox): add /api/public/ultravox-call endpoint and mount tools router"
```

---

## Task 6: Layer 1 persona system prompt

**Why:** Spec §5 — replace Claude's 501-line monolith with a ~80-line behavioral prompt suited to Ultravox's 8B model.

**Files:**
- Create: `data/system_prompt_ultravox.txt`

- [ ] **Step 6.1: Create the prompt file**

Create `data/system_prompt_ultravox.txt` with the following content:

```text
You are Shakira, the warm and knowledgeable AI advisor at {ACADEMY_NAME} in Dubai.

# ROLE
You help prospective students explore beauty therapy, spa management, massage, dermaplaning, and electrical facial courses. Your goal: answer their questions, then help them book a free counseling session with the admissions team.

# VOICE STYLE
- Speak naturally, like a real person on a phone call. NOT like a chatbot.
- Keep every reply under 25 spoken words unless reading a fact list.
- One topic per reply. Wait for their response.
- Warm, professional, never pushy.
- If asked, you are an AI assistant from EBH Academy. Do not pretend to be human.

# WHAT YOU CAN AND CANNOT SAY
- You CANNOT promise jobs, visas, or guaranteed outcomes.
- You CAN say "KHDA-accredited" and "CIDESCO-affiliated" — these are confirmed.
- DO NOT claim DHA license issuance — DHA licensing depends on the student.
- For anyone under 18, mention that parental consent is required for enrollment.
- If a caller sounds distressed or mentions self-harm: pause sales talk, express care, and share the UAE crisis line 800-HOPE (800-4673).

# HOW TO USE YOUR TOOLS
- For course info, pricing, duration, certifications: call search_courses with the topic.
- For accreditation, KHDA, DHA, CIDESCO, payment plans, refunds, location: call search_academy_faq.
- For current date or time: call get_current_datetime.
- Once you have the caller's NAME AND PHONE NUMBER, call capture_enrollment_lead.

# CONVERSATION FLOW
1. Greet them, ask what they're interested in.
2. Answer their question by calling the right tool first, then speaking the result conversationally.
3. After 3-4 turns OR when they show commitment ("I want to enroll", "sign me up"), ask: "May I have your name and phone number so admissions can call you with details?"
4. When you have both, call capture_enrollment_lead, then tell them admissions will follow up.
5. If they don't share contact info, that's fine — keep answering their questions.

# NUMBERS AND CURRENCY
- Prices are in AED (UAE Dirhams). Say "AED 5,500" as "five thousand five hundred dirhams".
- When reading dates, use the format "Monday May 20th".

# WHEN YOU DON'T KNOW
If the answer isn't in your tools, say:
"I don't have that detail, but our admissions team at +971 56 390 0330 can help — would you like me to pass your number to them?"

# LANGUAGE
If the caller starts in Arabic, reply in Arabic. If English, reply in English. Match their choice. Numbers, prices, and proper names stay readable in either language.
```

- [ ] **Step 6.2: Commit**

```bash
git add data/system_prompt_ultravox.txt
git commit -m "feat(ultravox): add Layer 1 persona system prompt for Ultravox agent"
```

> NOTE: This file is uploaded to the Ultravox agent dashboard at configuration time (Task 9). It is not loaded by Python at runtime.

---

## Task 7: Realtime web client page

**Why:** Web visitors need a page that opens a WebRTC connection to Ultravox using the joinUrl returned by `/api/public/ultravox-call`.

**Files:**
- Create: `demo/shakira_realtime.html`

- [ ] **Step 7.1: Create the file**

Create `demo/shakira_realtime.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Talk to Shakira (Realtime) | EBH Academy</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x1f393;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f4f0fa;
  --text: #1a1020;
  --muted: #7a6590;
  --accent: #7c3aed;
  --accent-deep: #5b21b6;
  --accent-light: rgba(124, 58, 237, 0.10);
  --line: rgba(100, 50, 150, 0.10);
  --panel: rgba(255, 255, 255, 0.84);
  --success: #2d7a50;
  --danger: #b5434b;
  --shadow: 0 20px 50px rgba(80, 30, 120, 0.10);
  --radius: 22px;
}
body {
  font-family: 'Outfit', system-ui, sans-serif;
  color: var(--text);
  background:
    radial-gradient(ellipse at 20% 0%, rgba(124, 58, 237, 0.15), transparent 50%),
    linear-gradient(180deg, #f8f5fd 0%, #ede6f7 100%);
  min-height: 100vh;
}
.container { max-width: 720px; margin: 0 auto; padding: 32px 16px; }
.brand { font-family: 'Cormorant', Georgia, serif; font-weight: 700; font-size: 1.8rem; color: var(--accent-deep); text-align: center; }
.tagline { color: var(--muted); text-align: center; margin-top: 4px; font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; }
.realtime-badge { display: inline-block; background: var(--accent-light); color: var(--accent-deep); padding: 4px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; margin-top: 8px; text-transform: uppercase; }
.panel { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--line); border-radius: var(--radius); padding: 28px; margin-top: 28px; box-shadow: var(--shadow); }
.orb {
  width: 160px; height: 160px; margin: 12px auto 24px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #b794ec, #7c3aed 60%, #5b21b6);
  box-shadow: 0 0 60px rgba(124, 58, 237, 0.4);
  transition: transform 0.3s, box-shadow 0.3s;
}
.orb.listening { animation: pulse 1.6s infinite ease-in-out; }
.orb.speaking { box-shadow: 0 0 80px rgba(124, 58, 237, 0.7); transform: scale(1.05); }
@keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.04); } }
.lang-picker { display: flex; gap: 8px; justify-content: center; margin-bottom: 20px; }
.lang-btn { background: white; border: 1px solid var(--line); padding: 8px 18px; border-radius: 999px; cursor: pointer; font-family: inherit; font-size: 0.9rem; transition: background 0.2s; }
.lang-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
.call-btn {
  display: block; margin: 0 auto; padding: 14px 36px;
  background: var(--accent); color: white; border: none; border-radius: 999px;
  font-family: inherit; font-size: 1rem; font-weight: 600; cursor: pointer;
  transition: background 0.2s, transform 0.2s;
}
.call-btn:hover { background: var(--accent-deep); transform: translateY(-1px); }
.call-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.call-btn.end { background: var(--danger); }
.status { text-align: center; color: var(--muted); margin-top: 16px; font-size: 0.85rem; min-height: 20px; }
.transcript {
  margin-top: 20px; max-height: 220px; overflow-y: auto;
  background: white; border: 1px solid var(--line); border-radius: 14px;
  padding: 14px; font-size: 0.92rem;
}
.t-row { margin-bottom: 8px; line-height: 1.5; }
.t-row.agent { color: var(--accent-deep); }
.t-row.user { color: var(--text); }
.t-role { font-weight: 600; margin-right: 6px; }
.classic-link { text-align: center; margin-top: 16px; font-size: 0.85rem; }
.classic-link a { color: var(--muted); text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
  <div class="brand">EBH Academy</div>
  <div class="tagline">Talk to Shakira</div>
  <div style="text-align: center;"><span class="realtime-badge">Realtime · Beta</span></div>

  <div class="panel">
    <div id="orb" class="orb"></div>

    <div class="lang-picker">
      <button class="lang-btn active" data-lang="en">English</button>
      <button class="lang-btn" data-lang="ar">العربية</button>
    </div>

    <button id="callBtn" class="call-btn">Start Call</button>
    <div id="status" class="status">Ready</div>

    <div id="transcript" class="transcript" style="display: none;"></div>
  </div>

  <div class="classic-link"><a href="/talk">Use classic version instead</a></div>
</div>

<script>
let lang = 'en';
let pc = null;
let micStream = null;
let dataChannel = null;
let inCall = false;

const orb = document.getElementById('orb');
const callBtn = document.getElementById('callBtn');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');

document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (inCall) return;
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    lang = btn.dataset.lang;
  });
});

callBtn.addEventListener('click', async () => {
  if (inCall) {
    endCall();
  } else {
    await startCall();
  }
});

async function startCall() {
  setStatus('Requesting microphone…');
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus('Microphone access denied.');
    return;
  }

  setStatus('Connecting…');
  let joinUrl;
  try {
    const res = await fetch(`/api/public/ultravox-call?lang=${lang}&source=web`);
    if (!res.ok) throw new Error(`gateway ${res.status}`);
    const json = await res.json();
    joinUrl = json.joinUrl;
    if (!joinUrl) throw new Error('no joinUrl');
  } catch (e) {
    setStatus('Could not start call. Try again.');
    cleanupMic();
    return;
  }

  try {
    await openWebRtc(joinUrl);
  } catch (e) {
    console.error(e);
    setStatus('Connection failed.');
    cleanupMic();
    return;
  }

  inCall = true;
  callBtn.textContent = 'End Call';
  callBtn.classList.add('end');
  orb.classList.add('listening');
  transcriptEl.style.display = 'block';
  transcriptEl.innerHTML = '';
  setStatus('Connected. Shakira is listening.');
}

async function openWebRtc(joinUrl) {
  pc = new RTCPeerConnection();
  micStream.getTracks().forEach(t => pc.addTrack(t, micStream));

  pc.ontrack = (ev) => {
    const audio = new Audio();
    audio.srcObject = ev.streams[0];
    audio.autoplay = true;
  };

  dataChannel = pc.createDataChannel('ultravox');
  dataChannel.onmessage = handleDataMessage;

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const resp = await fetch(joinUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/sdp' },
    body: offer.sdp,
  });
  if (!resp.ok) throw new Error(`signaling ${resp.status}`);
  const answer = await resp.text();
  await pc.setRemoteDescription({ type: 'answer', sdp: answer });
}

function handleDataMessage(ev) {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  if (msg.type === 'transcript' && msg.text) {
    appendTranscript(msg.role || 'agent', msg.text, msg.final !== false);
  } else if (msg.type === 'state') {
    if (msg.state === 'speaking') {
      orb.classList.add('speaking');
      orb.classList.remove('listening');
    } else if (msg.state === 'listening') {
      orb.classList.remove('speaking');
      orb.classList.add('listening');
    }
  }
}

function appendTranscript(role, text, isFinal) {
  const div = document.createElement('div');
  div.className = `t-row ${role}`;
  div.innerHTML = `<span class="t-role">${role === 'agent' ? 'Shakira' : 'You'}:</span>${text}`;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function endCall() {
  if (pc) { try { pc.close(); } catch {} pc = null; }
  cleanupMic();
  inCall = false;
  callBtn.textContent = 'Start Call';
  callBtn.classList.remove('end');
  orb.classList.remove('listening', 'speaking');
  setStatus('Call ended.');
}

function cleanupMic() {
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
}

function setStatus(msg) { statusEl.textContent = msg; }
</script>
</body>
</html>
```

- [ ] **Step 7.2: Smoke-check the page renders**

Start the gateway and visit the page:

```bash
uvicorn main:app --reload --port 8080
# In a browser: http://localhost:8080/talk-realtime
# Expected: page loads, orb visible, language picker works
# (Calling "Start" will fail without ULTRAVOX_* env vars set — that's fine)
```

> **Fallback if direct WebRTC fails in Task 10 smoke test:** Ultravox publishes a JS client SDK that wraps signaling. If the raw SDP-over-HTTP-POST exchange above doesn't connect, replace `openWebRtc()` with the official SDK. Load it from Ultravox's CDN per their docs (https://docs.ultravox.ai/) — the rest of the page (orb, transcript, button) stays the same. This is an explicit open question in the spec (§10.3).

- [ ] **Step 7.3: Commit**

```bash
git add demo/shakira_realtime.html
git commit -m "feat(ultravox): add realtime web client page (Talk to Shakira beta)"
```

---

## Task 8: Update `.env.example`

**Why:** New env vars must be documented for anyone running the project.

**Files:**
- Modify: `.env.example`

- [ ] **Step 8.1: Append the new section**

Append to the end of `.env.example`:

```bash
# ── Ultravox realtime stack (Phase 1, parallel to LiveKit) ───────────────
# Get API key from https://app.ultravox.ai/settings
ULTRAVOX_API_KEY=
# Agent UUID from your Ultravox dashboard
ULTRAVOX_AGENT_ID=
# Shared secret used to sign webhook callbacks (any long random string)
ULTRAVOX_TOOL_SECRET=
# Optional overrides
ULTRAVOX_BASE_URL=https://api.ultravox.ai
ULTRAVOX_MAX_DURATION=1800s
ULTRAVOX_AR_INWORLD_VOICE_ID=WcxyRPjVQcpVYmceBQO4Helb
```

- [ ] **Step 8.2: Commit**

```bash
git add .env.example
git commit -m "docs(env): document ULTRAVOX_* env vars for realtime stack"
```

---

## Task 9: Ultravox dashboard configuration (manual, one-time)

**Why:** Ultravox agents are created via their console, not via the API at every call. The `ULTRAVOX_AGENT_ID` env var points to this persistent agent record.

This task is manual setup done by the developer in the Ultravox web console at https://app.ultravox.ai/. No code changes; no commit.

- [ ] **Step 9.1: Sign up at https://app.ultravox.ai/ if not already**
- [ ] **Step 9.2: Settings → API Keys → create key → save to `.env` as `ULTRAVOX_API_KEY`**
- [ ] **Step 9.3: Generate a random `ULTRAVOX_TOOL_SECRET` (any 32+ char string) and add to `.env`**
- [ ] **Step 9.4: Agents → Create Agent. Paste `data/system_prompt_ultravox.txt` as the system prompt. Choose a warm female voice (test 2-3 candidates in Phase 2 internal testing).**
- [ ] **Step 9.5: Tools → add 4 tools, one per endpoint:**
  - `get_current_datetime` → POST `https://<your-ec2-host>/api/ultravox/tools/datetime` — no parameters
  - `search_courses` → POST `https://<your-ec2-host>/api/ultravox/tools/search-courses` — parameter `query: string`
  - `search_academy_faq` → POST `https://<your-ec2-host>/api/ultravox/tools/search-faq` — parameter `query: string`
  - `capture_enrollment_lead` → POST `https://<your-ec2-host>/api/ultravox/tools/capture-lead` — parameters `name: string`, `phone: string`, `email: string?`, `course_interest: string?`, `notes: string?`
- [ ] **Step 9.6: For each tool, attach the same `ULTRAVOX_TOOL_SECRET` as the shared secret used to sign requests**
- [ ] **Step 9.7: Copy the Agent UUID into `.env` as `ULTRAVOX_AGENT_ID`**

This task does not commit anything — the configuration lives on Ultravox's side and only the env vars (already gitignored) reference it.

---

## Task 10: End-to-end smoke test

**Why:** Make sure the parallel stack actually creates a call and tools fire.

This is a manual verification step that exercises the live stack with real Ultravox credentials.

- [ ] **Step 10.1: Set the new env vars locally**

In `.env` (gitignored), set the values from Task 9.

- [ ] **Step 10.2: Start the gateway**

```bash
uvicorn main:app --reload --port 8080
```

- [ ] **Step 10.3: Visit http://localhost:8080/talk-realtime**

Expected: page renders, language picker works, "Start Call" prompts for microphone permission.

- [ ] **Step 10.4: Click "Start Call"; speak: "What courses do you offer?"**

Expected on gateway logs:
- `Ultravox call created: callId=...`
- `Tool search_courses` log line as Shakira looks up the answer
- Audio response in browser

- [ ] **Step 10.5: Speak: "My name is Test User, my phone is 0561234567"**

Expected: `capture-lead` endpoint hits, session state moves to IDENTIFIED, log line shows `Enrollment lead captured`.

- [ ] **Step 10.6: End the call. Confirm classic stack still works**

Visit http://localhost:8080/talk — confirm the classic LiveKit Shakira loads. This page must be **unchanged**.

- [ ] **Step 10.7: Run the full test suite one more time**

```bash
pytest --basetemp=.pytest_tmp
# Expected: all green
```

- [ ] **Step 10.8: Push the branch**

```bash
git push -u origin feat/ultravox-realtime
```

Open a PR to `master`. Tag the PR with "Phase 1" and link the spec.

---

## Phase 1 Definition of Done

All checked:

- [ ] All unit tests pass (`pytest --basetemp=.pytest_tmp`)
- [ ] `/talk-realtime` loads and renders correctly
- [ ] `/api/public/ultravox-call?lang=en` returns a valid `joinUrl`
- [ ] All 4 tool endpoints reject requests without valid signatures (HTTP 401)
- [ ] All 4 tool endpoints accept signed requests and return expected payloads (HTTP 200)
- [ ] Lead capture transitions session state to IDENTIFIED
- [ ] Classic `/talk` page still works (no regression)
- [ ] `.env.example` documents all new vars
- [ ] PR opened to `master`

Phases 2 (internal testing), 3 (A/B), and 4 (cut-over) are out of scope for this plan.
