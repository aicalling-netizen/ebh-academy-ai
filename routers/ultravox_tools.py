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
    """Read body and verify the Ultravox signature. Return (body_bytes, call_id).

    Fast-fail order: secret missing → 503, headers missing → 401, then read
    body, then HMAC verify → 401 on mismatch.
    """
    secret = os.getenv("ULTRAVOX_TOOL_SECRET", "").strip()
    if not secret:
        logger.error("ULTRAVOX_TOOL_SECRET not configured — refusing webhook")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="server not configured")
    if not call_id or not timestamp or not signature:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing signature headers")
    body = await request.body()
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
