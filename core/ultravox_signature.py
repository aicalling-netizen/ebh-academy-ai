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
