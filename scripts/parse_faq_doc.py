"""Parse data/ebh_academy_faq.txt into structured JSON for the RAG pipeline.

Output: data/faq_knowledge.json containing:
  - courses: list of {name, accreditations, dha_eligible} from Part 3
  - faqs: list of {category, q, a} from Categories A-I (Parts 7-15)
  - principles: text block of Part 1 Honesty Principles + DHA lock
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "ebh_academy_faq.txt"
DST = ROOT / "data" / "faq_knowledge.json"


def _decode_entities(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def parse_course_matrix(text: str) -> list[dict]:
    """Parse Part 3 — Verified Course Certification Reference (table rows)."""
    # The table follows the header "Course Name CIDESCO KHDA IAO EBH DHA Eligible"
    # with one row per course over 5 lines: name, then 5 cells.
    m = re.search(
        r"Part 3 — Verified Course Certification Reference(.*?)(?:Part 4 —)", text, re.S
    )
    if not m:
        return []
    block = m.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and ln.strip() != "\xa0"]

    # Discard preamble until the first "DHA Eligible" header
    try:
        start = lines.index("DHA Eligible") + 1
    except ValueError:
        return []
    rows = lines[start:]

    courses: list[dict] = []
    i = 0
    cell_chars = {"✓", "—", "Yes", "No"}
    while i < len(rows):
        name = rows[i]
        # End markers
        if name.startswith("Key:") or name.startswith("Critical:") or name in cell_chars:
            break
        if i + 5 >= len(rows):
            break
        cidesco = rows[i + 1]
        khda = rows[i + 2]
        iao = rows[i + 3]
        ebh = rows[i + 4]
        dha = rows[i + 5]
        # Validate cells look like check/dash/Yes/No
        if all(c in cell_chars for c in (cidesco, khda, iao, ebh, dha)):
            courses.append(
                {
                    "name": name,
                    "cidesco": cidesco == "✓",
                    "khda": khda == "✓",
                    "iao": iao == "✓",
                    "ebh": ebh == "✓",
                    "dha_eligible": dha == "Yes",
                }
            )
            i += 6
        else:
            # Misaligned row — skip
            i += 1
    return courses


def parse_faq_categories(text: str) -> list[dict]:
    """Extract Q&A pairs from Categories A through I."""
    faqs: list[dict] = []
    # Each category starts with "Category X: Title" and ends at next "Category X:" or EOF.
    category_pattern = re.compile(
        r"Category\s+([A-Z]):\s*([^\n]+)\n(.*?)(?=Category\s+[A-Z]:|$)", re.S
    )
    for match in category_pattern.finditer(text):
        category_letter = match.group(1).strip()
        category_title = match.group(2).strip()
        body = match.group(3)
        # Q pattern: "Q1.  question text..." until next Q or category end.
        q_pattern = re.compile(
            r"Q(\d+)\.\s+(.+?)\n(.*?)(?=\nQ\d+\.\s+|\nCategory\s+[A-Z]:|$)", re.S
        )
        for q in q_pattern.finditer(body):
            q_num = int(q.group(1))
            q_text = q.group(2).strip()
            a_text = q.group(3).strip()
            # Normalise whitespace inside the answer
            a_text = re.sub(r"[ \t]+", " ", a_text)
            a_text = re.sub(r"\n{3,}", "\n\n", a_text).strip()
            faqs.append(
                {
                    "category": f"{category_letter} — {category_title}",
                    "q_num": q_num,
                    "q": q_text,
                    "a": a_text,
                }
            )
    return faqs


def parse_principles(text: str) -> str:
    m = re.search(r"Honesty Principles(.*?)(?:Information-First|Part 2 —)", text, re.S)
    return _decode_entities(m.group(1).strip()) if m else ""


def main() -> None:
    raw = SRC.read_text(encoding="utf-8")
    text = _decode_entities(raw)
    out = {
        "principles": parse_principles(text),
        "courses": parse_course_matrix(text),
        "faqs": parse_faq_categories(text),
    }
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {DST.name}: {len(out['courses'])} courses, "
        f"{len(out['faqs'])} FAQs, {len(out['principles'])}-char principles."
    )


if __name__ == "__main__":
    main()
