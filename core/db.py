"""Supabase client and database helpers for EBH Academy."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger("academy.db")

_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        logger.warning("Supabase not configured — database operations will be skipped")
        return None

    from supabase import create_client
    _supabase_client = create_client(url, key)
    logger.info("Supabase client initialized for %s", url)
    return _supabase_client


def normalize_phone(raw: str) -> str:
    """Normalize UAE phone numbers to +971XXXXXXXXX format."""
    digits = re.sub(r"[^\d+]", "", raw.strip())
    if digits.startswith("00971"):
        digits = "+971" + digits[5:]
    elif digits.startswith("971") and len(digits) >= 12:
        digits = "+" + digits
    elif digits.startswith("05") and len(digits) == 10:
        digits = "+971" + digits[1:]
    elif digits.startswith("5") and len(digits) == 9:
        digits = "+971" + digits
    elif not digits.startswith("+"):
        digits = "+" + digits
    return digits


async def save_inquiry(data: dict[str, Any]) -> dict[str, Any] | None:
    """Save a course inquiry / lead to the academy_inquiries table."""
    sb = _get_supabase()
    if sb is None:
        logger.warning("Skipping inquiry save — no Supabase client")
        return None

    try:
        result = sb.table("academy_inquiries").insert(data).execute()
        rows = getattr(result, "data", None) or []
        if rows:
            logger.info("Saved inquiry id=%s", rows[0].get("id"))
            return rows[0]
        return None
    except Exception:
        logger.exception("Failed to save inquiry")
        return None


async def get_inquiries(
    limit: int = 50,
    offset: int = 0,
    phone: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve academy inquiries, newest first."""
    sb = _get_supabase()
    if sb is None:
        return []

    try:
        q = sb.table("academy_inquiries").select("*").order("created_at", desc=True)
        if phone:
            q = q.eq("phone", normalize_phone(phone))
        q = q.range(offset, offset + limit - 1)
        result = q.execute()
        return getattr(result, "data", []) or []
    except Exception:
        logger.exception("Failed to fetch inquiries")
        return []


async def save_call_log(data: dict[str, Any]) -> dict[str, Any] | None:
    """Save a call log entry."""
    sb = _get_supabase()
    if sb is None:
        return None

    try:
        result = sb.table("academy_call_logs").insert(data).execute()
        rows = getattr(result, "data", None) or []
        return rows[0] if rows else None
    except Exception:
        logger.exception("Failed to save call log")
        return None


async def get_call_logs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    sb = _get_supabase()
    if sb is None:
        return []

    try:
        result = (
            sb.table("academy_call_logs")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return getattr(result, "data", []) or []
    except Exception:
        logger.exception("Failed to fetch call logs")
        return []
