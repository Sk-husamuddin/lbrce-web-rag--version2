"""Structured parser for official LBRCE student roster pages.

Unlike the generic HTML parser, this module walks roster tables row by row.
It emits complete pipe-delimited rows together with the cohort and section
heading that precedes each table, matching the consumer contract in
``backend.graph.nodes``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from backend.ingestion.chunker import Chunk


YEAR_TO_SEMESTER = {"II": "III", "III": "V", "IV": "VII"}
COHORT_TO_ACADEMIC_YEAR = {
    "2025": "2026-27",
    "2024": "2026-27",
    "2023": "2026-27",
}

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_YEAR_HEADING_RE = re.compile(
    r"\b(?:(?P<batch>(?:19|20)\d{2})\s+Batch\s*-\s*)?"
    r"(?P<year>II|III|IV)\s+Year\b",
    re.IGNORECASE,
)
_BATCH_HEADING_RE = re.compile(
    r"\b(?P<batch>20\d{2})\s+Batch\s*-\s*(?P<year>II|III|IV)\s+Year\b",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"\b(?:section|sec\.?)\s*[-:]?\s*(?P<section>[A-H])\b"
    r"|\b(?P<section_slash>[A-H])\s*/\s*sec\.?\b"
    r"|\b(?P<section_word>[A-H])\s+section\b",
    re.IGNORECASE,
)
_SERIAL_RE = re.compile(r"^\s*\d+\s*$")
_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{4,}$", re.IGNORECASE)


@dataclass(frozen=True)
class RosterTable:
    """A single official roster table with its preceding labels and rows."""

    cohort_heading: str
    section_heading: str
    rows: List[str]
    semester: str = ""
    academic_year: str = "2026-27"
    section: str = ""


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "roster"


def _heading_label(text: str) -> Optional[tuple[str, str, str]]:
    """Return normalized cohort heading, semester, and academic year."""
    text = _clean(text)
    match = _YEAR_HEADING_RE.search(text)
    if not match:
        return None
    year = match.group("year").upper()
    batch_match = _BATCH_HEADING_RE.search(text)
    batch = batch_match.group("batch") if batch_match else ""
    if batch:
        heading = text
    else:
        heading = f"{year} Year Students List"
    semester = YEAR_TO_SEMESTER.get(year, "")
    academic_year = COHORT_TO_ACADEMIC_YEAR.get(batch, "2026-27")
    return heading, semester, academic_year


def _section_from_text(text: str) -> str:
    match = _SECTION_RE.search(_clean(text))
    if not match:
        return ""
    return (
        match.group("section")
        or match.group("section_slash")
        or match.group("section_word")
        or ""
    ).upper()


def _looks_like_section_label(text: str) -> bool:
    lowered = _clean(text).lower()
    return bool(
        _section_from_text(text)
        or "semester" in lowered
        or re.search(r"\b(?:i|ii|iii|iv|v|vi|vii)\s*sem\.?\b", lowered)
        or re.search(r"\b(?:i|ii|iii|iv)\s*year\b", lowered)
    )


def _serialize_row(cells: List[str]) -> Optional[str]:
    """Serialize a table row as serial | registration number | name."""
    if len(cells) < 3 or not _SERIAL_RE.match(cells[0]):
        return None

    serial = str(int(cells[0]))
    registration = ""
    for candidate in cells[1:-1]:
        candidate = _clean(candidate)
        if _ID_RE.match(candidate) and any(char.isdigit() for char in candidate):
            registration = candidate
            break
    if not registration:
        registration = _clean(cells[1])
    name = _clean(cells[-1])
    if not registration or not name:
        return None
    return f"{serial} | {registration} | {name}"


def _table_rows(table: Any) -> List[str]:
    rows: List[str] = []
    for row in table.find_all("tr"):
        cells = [_clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        serialized = _serialize_row(cells)
        if serialized:
            rows.append(serialized)
    return rows


def _table_section_heading(table: Any) -> str:
    """Get the section label that is normally the table's first non-data row."""
    for row in table.find_all("tr"):
        cells = [_clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if not cells:
            continue
        if _serialize_row(cells):
            continue
        joined = " | ".join(cells)
        if _looks_like_section_label(joined):
            return joined
    return ""


def _iter_tables_with_context(soup: BeautifulSoup) -> Iterable[tuple[Any, str, str, str]]:
    """Yield each table with the latest cohort heading and inferred labels."""
    current_cohort = ""
    current_semester = ""
    current_academic_year = "2026-27"
    for element in soup.find_all(list(_HEADING_TAGS) + ["table"]):
        if element.name in _HEADING_TAGS:
            parsed = _heading_label(element.get_text(" ", strip=True))
            if parsed:
                current_cohort, current_semester, current_academic_year = parsed
            continue
        yield element, current_cohort, current_semester, current_academic_year


def extract_roster_tables(html: str) -> List[RosterTable]:
    """Extract all department roster tables without flattening their rows."""
    soup = BeautifulSoup(html or "", "html.parser")
    tables: List[RosterTable] = []
    for table, cohort, semester, academic_year in _iter_tables_with_context(soup):
        rows = _table_rows(table)
        if not rows:
            continue
        section_heading = _table_section_heading(table)
        section = _section_from_text(section_heading)
        # Some pages expose a section heading immediately before the table
        # instead of inside the first row. Use the closest matching heading.
        if not section_heading:
            previous = table.find_all_previous(_HEADING_TAGS, limit=8)
            for heading in previous:
                candidate = _clean(heading.get_text(" ", strip=True))
                if _looks_like_section_label(candidate):
                    section_heading = candidate
                    section = _section_from_text(candidate)
                    break
        if not cohort:
            fallback = _clean(section_heading)
            parsed = _heading_label(fallback)
            if parsed:
                cohort, semester, academic_year = parsed
        if not cohort:
            cohort = "Official LBRCE Students List"
        if not semester:
            match = re.search(r"\b(II|III|IV)\s*Year\b", cohort, re.IGNORECASE)
            if match:
                semester = YEAR_TO_SEMESTER[match.group(1).upper()]
        tables.append(
            RosterTable(
                cohort_heading=cohort,
                section_heading=section_heading,
                rows=rows,
                semester=semester,
                academic_year=academic_year,
                section=section,
            )
        )
    return tables


def _directory_chunk(url: str, title: str, department: str, html: str) -> Chunk:
    soup = BeautifulSoup(html or "", "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href"))
        label = _clean(anchor.get_text(" ", strip=True))
        if "studentslist.php" in href.lower():
            links.append(f"{label or href} | {href}")
    text = "Official LBRCE Department Wise Students List\n\n" + "\n".join(dict.fromkeys(links))
    return Chunk(
        chunk_id=f"student-list-directory-{_slug(url)}",
        text=text.strip(),
        source_url=url,
        title=title,
        source_type="student_list_html",
        department=department,
        page_number=0,
        document_id=f"student-list-{_slug(url)}",
        metadata={
            "resource_type": "student_list_html",
            "source_type": "student_list_html",
            "department": department,
            "source_url": url,
            "title": title,
        },
    )


def parse_student_list_html(
    html: str,
    *,
    url: str,
    title: str,
    department: str = "",
    chunk_size: int = 500,
) -> List[Chunk]:
    """Return row-safe Pinecone chunks for one approved roster page."""
    tables = extract_roster_tables(html)
    if not tables:
        return [_directory_chunk(url, title, department, html)]

    chunks: List[Chunk] = []
    document_id = f"student-list-{department or _slug(url)}"
    max_chars = max(800, chunk_size * 4)
    for table_index, roster in enumerate(tables):
        context = [roster.cohort_heading]
        if roster.section_heading:
            context.append(roster.section_heading)
        context.append("S. No. | Regd. Num. | Name of the Student")
        rows: List[str] = []
        chunk_part = 0
        for row in roster.rows:
            candidate = "\n".join(context + rows + [row])
            if rows and len(candidate) > max_chars:
                text = "\n".join(context + rows)
                metadata = {
                    "resource_type": "student_list_html",
                    "source_type": "student_list_html",
                    "source_url": url,
                    "title": title,
                    "department": department,
                    "academic_year": roster.academic_year,
                    "semester": roster.semester,
                    "section": roster.section,
                    "student_list_cohort": roster.cohort_heading,
                    "student_list_section": roster.section_heading,
                }
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}-{table_index}-{chunk_part}",
                        text=text,
                        source_url=url,
                        title=title,
                        source_type="student_list_html",
                        department=department,
                        page_number=0,
                        document_id=document_id,
                        metadata=metadata,
                    )
                )
                chunk_part += 1
                rows = []
            rows.append(row)
        if rows:
            text = "\n".join(context + rows)
            metadata = {
                "resource_type": "student_list_html",
                "source_type": "student_list_html",
                "source_url": url,
                "title": title,
                "department": department,
                "academic_year": roster.academic_year,
                "semester": roster.semester,
                "section": roster.section,
                "student_list_cohort": roster.cohort_heading,
                "student_list_section": roster.section_heading,
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-{table_index}-{chunk_part}",
                    text=text,
                    source_url=url,
                    title=title,
                    source_type="student_list_html",
                    department=department,
                    page_number=0,
                    document_id=document_id,
                    metadata=metadata,
                )
            )
    return chunks


async def fetch_student_list_html(url: str, *, timeout: float = 30.0, user_agent: str = "LBRCE-Roster-Ingestion/1.0") -> str:
    """Fetch one approved HTTPS LBRCE roster page."""
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"lbrce.ac.in", "www.lbrce.ac.in"}:
        raise ValueError(f"URL is outside the approved LBRCE domain: {url}")
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": user_agent}) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
