"""Tool: search EBH Academy FAQ for accreditation, policies, and general questions."""
from __future__ import annotations

import logging

from core.rag_pipeline import search_faq

logger = logging.getLogger("academy.tools.faq")


async def _handle(arguments: dict) -> dict:
    """Search the academy FAQ knowledge base."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"status": "error", "message": "Please provide a search query."}

    results = search_faq(query, max_results=3)
    logger.info("FAQ search for '%s' returned %d results", query, len(results))

    if not results:
        return {
            "status": "no_results",
            "message": "I don't have specific information on that. Please contact our admissions team at support@ebhacademy.com or call +971 56 390 0330.",
        }

    return {
        "status": "ok",
        "count": len(results),
        "results": [{"question": r["q"], "answer": r["a"]} for r in results],
    }
