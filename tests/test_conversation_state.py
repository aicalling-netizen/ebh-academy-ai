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
