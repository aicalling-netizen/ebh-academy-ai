"""RAG pipeline for EBH Academy course content and FAQs.

Builds a FAISS vector index over structured course data and policy documents.
Falls back to keyword matching when the embedding model isn't loaded.
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

_CACHE_DIR = Path(__file__).parent.parent / "data" / "rag_cache"
_SOURCES_DIR = Path(__file__).parent.parent / "data" / "rag_sources"

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


def search_faq(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Simple keyword search over built-in FAQ data."""
    query_lower = query.lower()
    scored: list[tuple[int, dict]] = []

    for item in ACCREDITATION_FAQ:
        score = 0
        for word in query_lower.split():
            if word in item["q"].lower():
                score += 3
            if word in item["a"].lower():
                score += 1
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_results]]
