"""RAG pipeline for EBH Academy course content and FAQs.

Loads structured knowledge from data/faq_knowledge.json (parsed from the official
EBH Academy FAQ document by scripts/parse_faq_doc.py). Falls back to in-code
constants when the JSON file is missing (test environments, fresh checkouts).

Search uses keyword scoring — good enough for ~165 Q&As. Upgrade to embeddings
later if recall becomes a bottleneck.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("academy.rag")

_INDEX = None
_CHUNKS: list[dict[str, str]] = []
_EMBEDDER = None

_DATA_DIR = Path(__file__).parent.parent / "data"
_CACHE_DIR = _DATA_DIR / "rag_cache"
_SOURCES_DIR = _DATA_DIR / "rag_sources"
_KNOWLEDGE_FILE = _DATA_DIR / "faq_knowledge.json"


def _load_knowledge() -> dict[str, Any]:
    """Read the parsed FAQ JSON; return empty dict if unavailable."""
    if not _KNOWLEDGE_FILE.exists():
        logger.warning("faq_knowledge.json not found at %s — using in-code fallback", _KNOWLEDGE_FILE)
        return {}
    try:
        return json.loads(_KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load faq_knowledge.json: %s — using in-code fallback", e)
        return {}


_KB = _load_knowledge()

# Full FAQ corpus from the parsed knowledge file (Categories A through I).
# Each entry: {category, q_num, q, a}.
ACADEMY_FAQ: list[dict[str, Any]] = _KB.get("faqs", []) or []

# Course Certification Matrix from Part 3 of the FAQ doc.
# Each entry: {name, cidesco, khda, iao, ebh, dha_eligible}.
COURSE_MATRIX: list[dict[str, Any]] = _KB.get("courses", []) or []

# Operating principles block (Honesty rules, DHA lock, FOMO discipline).
ACADEMY_PRINCIPLES: str = _KB.get("principles", "") or ""

# ── Built-in course knowledge (always available, no external docs needed) ──
COURSES: list[dict[str, Any]] = [
    {
        "name": "CIDESCO Diploma in Beauty Therapy",
        "duration": "1 Year",
        "price_aed": 24_000,
        "level": "Comprehensive (beginner to advanced)",
        "language": "English",
        "prerequisites": "None — open to beginners and professionals",
        "certifications": [
            "EBH Academy Certificate",
            "KHDA attestation",
            "CIDESCO Diploma (Switzerland)",
            "IAO Certificate",
            "DHA license eligibility",
        ],
        "curriculum_highlights": [
            "Anatomy, physiology, pathology",
            "Skin analysis (Fitzpatrick, Woods Lamp)",
            "Facial skincare, makeup, manicure/pedicure",
            "Micro-needling, microdermabrasion, chemical peels",
            "Radio frequency, electrotherapy",
            "Client consultation and management",
        ],
        "payment_notes": "Tabby installments available",
        "career_outcome": "Positions graduates for careers across the UAE beauty sector with internationally recognized credentials",
    },
    {
        "name": "CIDESCO Diploma in Beauty and Spa Management",
        "duration": "3 Months",
        "price_aed": 8_999,
        "level": "Professional",
        "language": "English",
        "prerequisites": "None — open to beginners and professionals",
        "certifications": [
            "EBH Academy Certificate",
            "KHDA attestation",
            "CIDESCO Diploma",
            "IAO Certificate",
        ],
        "curriculum_highlights": [
            "International/national beauty trends and technology",
            "Planning and operational management",
            "Leadership and professional development",
            "Stock control and human resources",
            "Finance, marketing, and customer relations",
            "Business plan development",
        ],
        "assessment": "MCQ + PowerPoint presentation + business plan project",
        "payment_notes": "Tabby installments available",
        "career_outcome": "Spa/salon management roles with internationally recognized credentials",
    },
    {
        "name": "Madero Body Massage Course",
        "duration": "3 Days",
        "price_aed": 3_999,
        "level": "Advanced",
        "language": "English",
        "prerequisites": "None — beginners and professionals welcome",
        "certifications": [
            "EBH Academy Certificate",
            "KHDA attestation",
            "International recognition credentials",
        ],
        "curriculum_highlights": [
            "Maderotherapy history and benefits",
            "Body anatomy for massage",
            "Professional 11-piece Madero tool set included",
            "Anti-cellulite theory and techniques",
            "Body contouring, lymphatic activation",
            "Client consultation, health & safety, aftercare",
        ],
        "payment_notes": "Tabby installments available",
        "career_outcome": "Specialist body massage therapist",
    },
    {
        "name": "Diploma in Body Therapy (Dermaplaning)",
        "duration": "1 Day",
        "price_aed": 1_799,
        "level": "Professional",
        "language": "English",
        "prerequisites": "None — beginners and professionals welcome",
        "certifications": [
            "EBH Academy Elite Certificate",
            "KHDA attestation",
            "IAO Certificate",
        ],
        "curriculum_highlights": [
            "Skin anatomy and classification",
            "Dermaplaning procedures using surgical scalpels",
            "Benefits, contraindications, aftercare",
            "Client consultation and preparation",
            "Practice on dummies and live models",
        ],
        "payment_notes": "Tabby installments available",
        "career_outcome": "Certified dermaplaning specialist",
    },
    {
        "name": "Electrical Facial Course",
        "duration": "2 Days",
        "price_aed": 1_499,
        "level": "Advanced",
        "language": "English",
        "prerequisites": "None — suitable for beginners and professionals",
        "certifications": [
            "EBH Academy Certificate",
            "IAO Certificate",
            "CIDESCO attestation",
        ],
        "curriculum_highlights": [
            "Galvanic facials, ultrasound facials",
            "High frequency facials",
            "Diamond peeling, vacuum lymphatic drainage",
            "Skin physiology and client assessment",
            "Hands-on training on dummies and live models",
        ],
        "payment_notes": "Contact for payment options",
        "career_outcome": "Advanced facial treatment specialist",
    },
]

ACCREDITATION_FAQ = [
    {
        "q": "What is KHDA?",
        "a": "KHDA (Knowledge and Human Development Authority) is the UAE government body that regulates private education in Dubai. Our certificates are KHDA-attested.",
    },
    {
        "q": "What is CIDESCO?",
        "a": "CIDESCO is the world's most prestigious international beauty therapy qualification, recognized in over 40 countries. It originated in Switzerland.",
    },
    {
        "q": "Can I get a DHA license after completing the course?",
        "a": "Yes, our CIDESCO Diploma in Beauty Therapy supports DHA (Dubai Health Authority) license eligibility, which is required to practice in Dubai clinics.",
    },
    {
        "q": "What is IAO?",
        "a": "IAO (International Accreditation Organization) provides international recognition for educational institutions and their certificates.",
    },
    {
        "q": "Do I need experience to enroll?",
        "a": "No, all our courses are open to beginners with no prior experience as well as working professionals looking to upskill.",
    },
    {
        "q": "What payment options are available?",
        "a": "We offer Tabby installment payments (buy now, pay later) on most courses. Contact us for specific installment plans.",
    },
    {
        "q": "Where is EBH Academy located?",
        "a": "We are at 117, Block B, Al Hudaiba Awards Buildings, Jumeirah 1, Dubai, UAE.",
    },
    {
        "q": "Is training hands-on?",
        "a": "Yes, all courses include hands-on practice on both practice dummies and live models in our in-house clinic.",
    },
    {
        "q": "Do you offer online courses?",
        "a": "Currently all our courses are in-person at our Dubai campus. This ensures you get the hands-on clinical training that employers value.",
    },
    {
        "q": "What is the refund policy?",
        "a": "We have a formal Refund & Compensation Policy. Please contact us at support@ebhacademy.com for specific details about refund terms.",
    },
]


def search_courses(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Simple keyword search over built-in course data."""
    query_lower = query.lower()
    scored: list[tuple[int, dict]] = []

    for course in COURSES:
        score = 0
        name_lower = course["name"].lower()
        if query_lower in name_lower:
            score += 10
        for word in query_lower.split():
            if word in name_lower:
                score += 3
            for highlight in course["curriculum_highlights"]:
                if word in highlight.lower():
                    score += 1
            for cert in course["certifications"]:
                if word in cert.lower():
                    score += 2
        if score > 0:
            scored.append((score, course))

    if not scored:
        return COURSES[:max_results]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_results]]


def search_faq(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Keyword search over the full FAQ corpus from data/faq_knowledge.json.

    Falls back to the legacy ACCREDITATION_FAQ list if the JSON wasn't loaded.
    Each returned item has at least q and a fields.
    """
    pool: list[dict[str, Any]] = ACADEMY_FAQ or ACCREDITATION_FAQ
    if not pool:
        return []
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) >= 2]
    if not query_words:
        return pool[:max_results]

    scored: list[tuple[int, dict]] = []
    for item in pool:
        q = item.get("q", "").lower()
        a = item.get("a", "").lower()
        score = 0
        # Exact phrase match in question is gold
        if query_lower in q:
            score += 20
        for word in query_words:
            if word in q:
                score += 5
            if word in a:
                score += 1
        if score > 0:
            scored.append((score, item))

    if not scored:
        # No keyword hits — return first N as a generic-info fallback
        return pool[:max_results]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_results]]


def get_course_credentials(course_name: str) -> dict[str, Any] | None:
    """Look up the accreditation matrix row for a given course name.

    Returns {cidesco, khda, iao, ebh, dha_eligible} or None if not found.
    Case- and partial-match tolerant.
    """
    if not COURSE_MATRIX:
        return None
    needle = course_name.lower().strip()
    # Try exact, then substring, then word-overlap
    for entry in COURSE_MATRIX:
        if entry["name"].lower() == needle:
            return entry
    for entry in COURSE_MATRIX:
        if needle in entry["name"].lower() or entry["name"].lower() in needle:
            return entry
    return None
