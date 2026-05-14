"""UAE timezone helpers for EBH Academy agent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

UAE_TZ = timezone(timedelta(hours=4))


def now_uae() -> datetime:
    return datetime.now(UAE_TZ)


def build_uae_time_context() -> str:
    now = now_uae()
    return (
        f"Current UAE date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')} (Gulf Standard Time, UTC+4). "
        f"Day of week: {now.strftime('%A')}."
    )
