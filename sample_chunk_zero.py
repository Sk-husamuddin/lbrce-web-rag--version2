"""Sample chunk_index=0 across the new corpus and flag nav-diluted chunks.

Run locally against the migration copy. No Pinecone connection needed.

Usage:
    python sample_chunk_zero.py --chunks migration_artifacts/full_registry_local_bge/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Same navigation labels used in assess_page_quality(), plus a few common
# short nav fragments that show up mid-line rather than as a whole line.
NAV_MARKERS = {
    "home", "administration", "lbrce trust", "founder chairman",
    "honorary chairman", "chairman", "vice-chairman", "president",
    "principal", "vice-principal", "governing body", "academic council",
    "board of studies (bos)", "finance committee", "strategic plan",
    "organisation structure", "dean academics", "academic regulations",
    "academic calendars", "lesson plans", "course structure & syllabus",
    "students list", "timetables", "faculty", "service rules", "rti act",
    "careers", "college brochure", "contact", "quick links",
    "mandatory disclosure", "aicte approvals", "ugc recognitions",
    "jntu affiliations", "nba", "nirf", "financial audit statements",
    "webmail", "gallery", "alumni", "placements", "cafeteria", "hostels",
    "central library", "programmes offered", "admission procedure",
    "fee structure",
}


def nav_ratio(text: str) -> float:
    """Fraction of non-empty lines that look like navigation-menu fragments."""
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    nav_lines = sum(1 for line in lines if line in NAV_MARKERS)
    return nav_lines / len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--nav-ratio-threshold", type=float, default=0.35,
                         help="Flag chunks where more than this fraction of lines look like nav")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_zeros = [c for c in chunks if c.get("chunk_index") == 0]
    print(f"Total chunks: {len(chunks)} | chunk_index=0 entries: {len(chunk_zeros)}")

    scored = []
    for chunk in chunk_zeros:
        ratio = nav_ratio(chunk.get("text", ""))
        scored.append((ratio, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    flagged = [pair for pair in scored if pair[0] > args.nav_ratio_threshold]

    print(f"\nFlagged (nav_ratio > {args.nav_ratio_threshold}): {len(flagged)} / {len(chunk_zeros)}\n")
    for ratio, chunk in flagged[:30]:
        print(f"  nav_ratio={ratio:.2f}  page_category={chunk.get('page_category')}  "
              f"topic={chunk.get('topic')}  url={chunk.get('source_url')}")

    random.seed(args.seed)
    remaining = [pair for pair in scored if pair not in flagged]
    sample = random.sample(remaining, min(args.sample_size, len(remaining)))

    print(f"\n--- Random sample of {len(sample)} unflagged chunk_index=0 entries for manual read ---\n")
    for ratio, chunk in sample:
        print(f"### nav_ratio={ratio:.2f} | {chunk.get('page_category')}/{chunk.get('topic')} | {chunk.get('source_url')}")
        print(chunk.get("text", "")[:400])
        print("...\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
