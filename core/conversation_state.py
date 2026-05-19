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
