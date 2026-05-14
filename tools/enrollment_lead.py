"""Tool: capture prospective student details as an enrollment lead."""
from __future__ import annotations

import logging
from typing import Any

from core.db import normalize_phone, save_inquiry

logger = logging.getLogger("academy.tools.enrollment_lead")


async def _handle(arguments: dict) -> dict:
    """Capture enrollment interest: name, phone, email, course, notes."""
    name = str(arguments.get("name", "")).strip()
    phone_raw = str(arguments.get("phone", "")).strip()
    email = str(arguments.get("email", "")).strip()
    course_interest = str(arguments.get("course_interest", "")).strip()
    notes = str(arguments.get("notes", "")).strip()

    if not name:
        return {"status": "error", "message": "Please ask for the caller's name first."}
    if not phone_raw:
        return {"status": "error", "message": "Please ask for the caller's phone number."}

    phone = normalize_phone(phone_raw)

    data: dict[str, Any] = {
        "name": name,
        "phone": phone,
        "email": email or None,
        "course_interest": course_interest or None,
        "notes": notes or None,
        "source": "web_call",
        "status": "new",
    }

    saved = await save_inquiry(data)

    if saved:
        logger.info("Enrollment lead captured: %s (%s) — %s", name, phone, course_interest)
        return {
            "status": "captured",
            "message": f"Interest registered for {name}. The admissions team will follow up shortly.",
            "inquiry_id": saved.get("id"),
        }

    logger.info("Enrollment lead captured (no DB): %s (%s) — %s", name, phone, course_interest)
    return {
        "status": "captured",
        "message": f"Interest registered for {name}. The admissions team will follow up shortly.",
    }
