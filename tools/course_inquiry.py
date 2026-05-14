"""Tool: search EBH Academy courses by keyword or interest."""
from __future__ import annotations

import logging
from typing import Any

from core.rag_pipeline import search_courses

logger = logging.getLogger("academy.tools.course_inquiry")


def _format_course(course: dict[str, Any]) -> dict[str, Any]:
    """Format a course record for the LLM."""
    return {
        "name": course["name"],
        "duration": course["duration"],
        "price": f"AED {course['price_aed']:,}",
        "level": course.get("level", ""),
        "language": course.get("language", "English"),
        "prerequisites": course.get("prerequisites", "None"),
        "certifications": course.get("certifications", []),
        "highlights": course.get("curriculum_highlights", []),
        "payment": course.get("payment_notes", ""),
        "career_outcome": course.get("career_outcome", ""),
    }


async def _handle(arguments: dict) -> dict:
    """Search courses matching the caller's interest."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        query = "all courses"

    results = search_courses(query, max_results=3)
    logger.info("Course search for '%s' returned %d results", query, len(results))

    if not results:
        return {
            "status": "no_results",
            "message": "No courses matched that query. We offer 5 professional courses in beauty therapy, spa management, body massage, dermaplaning, and electrical facials.",
        }

    return {
        "status": "ok",
        "count": len(results),
        "courses": [_format_course(c) for c in results],
    }
