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

    def test_known_vector_matches(self):
        # Pre-computed HMAC-SHA256 of "1.c.x" with secret "s".
        # External ground truth — protects against accidental changes to the
        # message construction formula (delimiter, byte order, etc.) by NOT
        # being derived from the implementation under test.
        known_sig = "b42768fdc63853802e4477b7cb9837752dc2a876f7a89796b948e270b63af8c0"
        assert verify_signature(
            body=b"x",
            timestamp="1",
            call_id="c",
            signature=known_sig,
            secret="s",
        ) is True
