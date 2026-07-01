"""Academy RDS dual-write — mirror enrolments/calls into `academy_calling`
(the database dashboard-v2 reads).

Supabase stays the PRIMARY store; this is a best-effort mirror for the dashboard.
Gated by DUAL_WRITE_MYSQL. Never raises — a mirror failure must not break a call.
Fully isolated: only ever writes to the academy DB (PAM_MYSQL_DATABASE), never the
clinic's `ai_calling`.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("academy.mysql_writer")


def _enabled() -> bool:
    return os.getenv("DUAL_WRITE_MYSQL", "").strip().lower() in {"1", "true", "yes", "on"}


def _conn():
    import pymysql
    return pymysql.connect(
        host=(os.getenv("PAM_MYSQL_HOST") or os.getenv("DB_HOST") or "").strip(),
        port=int((os.getenv("PAM_MYSQL_PORT") or os.getenv("DB_PORT") or "3306").strip() or 3306),
        user=(os.getenv("PAM_MYSQL_USER") or os.getenv("DB_USER") or "").strip(),
        password=(os.getenv("PAM_MYSQL_PASSWORD") or os.getenv("DB_PASSWORD") or ""),
        database=(os.getenv("PAM_MYSQL_DATABASE") or os.getenv("DB_NAME") or "academy_calling").strip(),
        connect_timeout=6,
        autocommit=True,
    )


def mirror_enrollment(
    *,
    name: str,
    phone: str,
    email: str = "",
    course_interest: str = "",
    notes: str = "",
    direction: str = "inbound",
    disposition: str = "inquiry",
    duration_seconds: int = 0,
) -> None:
    """Mirror one enrolment as a contact + call row in academy_calling.

    No-op unless DUAL_WRITE_MYSQL is on. Never raises (logs and returns on error).
    """
    if not _enabled():
        return
    try:
        first, _, last = (name or "").strip().partition(" ")
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO contacts "
                    "(first_name, last_name, phone_number, normalized_phone, email, status, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, 'active', NOW(), NOW())",
                    (first or name or None, last or None, phone or None, phone or None, email or None),
                )
                contact_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO calls "
                    "(call_uuid, contact_id, direction, channel_type, status, disposition, "
                    " started_at, ended_at, duration_seconds, metadata, created_at, updated_at) "
                    "VALUES (UUID(), %s, %s, 'ai', 'completed', %s, NOW(), NOW(), %s, %s, NOW(), NOW())",
                    (
                        contact_id, direction, disposition, int(duration_seconds or 0),
                        json.dumps({"course_interest": course_interest, "notes": notes, "source": "web-realtime"}),
                    ),
                )
            logger.info("RDS mirror: enrolment '%s' -> academy_calling (contact_id=%s)", name, contact_id)
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover — best-effort mirror
        logger.warning("RDS mirror failed (non-fatal, Supabase is source of truth): %r", e)
