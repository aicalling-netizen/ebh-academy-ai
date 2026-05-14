"""Tool: get the current date/time in UAE timezone."""
from __future__ import annotations

import logging

from core.time_context import now_uae

logger = logging.getLogger("academy.tools.datetime")


async def _handle(arguments: dict) -> dict:
    """Return current UAE date and time."""
    now = now_uae()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%I:%M %p"),
        "day_of_week": now.strftime("%A"),
        "timezone": "Gulf Standard Time (UTC+4)",
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p GST"),
    }
