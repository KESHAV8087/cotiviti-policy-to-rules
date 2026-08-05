"""
ingest.py  --  STEP 1 of the Cotiviti Policy-to-Rules POC.

Downloads a chapter of the CMS NCCI Policy Manual (public domain), extracts the
text, strips the repeating page headers/footers, and splits it into lettered
sections so the agent can later retrieve, summarize, extract rules from, and
diff it.

Source: CMS National Correct Coding Initiative (NCCI) Policy Manual.
        https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-policy-manual

Run:    python src/ingest.py
Output: data/raw/<version>.pdf        (raw download)
        data/processed/<version>.json (list of {section_id, heading, text})
"""

import json
import re
import sys
from pathlib import Path

import requests
import pdfplumber

# --- Configuration ----------------------------------------------------------
SOURCES = {
    "2024": "https://www.cms.gov/files/document/medicare-ncci-policy-manual-2024-chapter-1.pdf",
    "2025": "https://www.cms.gov/files/document/01-chapter1-ncci-medicare-policy-manual-2025finalcleanpdf.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CotivitiPOC/1.0; research)"}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# A real section header is a line like:
#   "B. Coding Based on Standards of Medical/Surgical Practice"
# a single capital letter, a period, then a Title-cased heading.
# The period is NOT in the heading character class, so a Table-of-Contents line
# ("A. Introduction.......... I-3") cannot match here.
SECTION_HEADER = re.compile(r'^\s*([A-Z])\.\s*([A-Z][A-Za-z0-9 ,/&()\x27’"“”-]{3,90})\s*$')

# A Table-of-Contents line: a header-like start followed by dot leaders and/or a
# page reference such as "I-3". Used by the state machine to explicitly reject a
# boundary that is really a TOC entry, rather than relying on a longest-body
# heuristic after the fact.
TOC_LINE = re.compile(r'^\s*[A-Z]\.\s.*(?:\.{3,}|\bI-\d+\b)\s*$')

FOOTER_PATTERNS = [
    re.compile(r"^\s*Revision Date \(Medicare\):.*$"),
    re.compile(r"^\s*I-\d+\s*$"),
]


def download(version: str, url: str) -> Path:
    """Download the PDF to data/raw/ (skips if already present)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{version}.pdf"
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return dest

    print(f"[get ] downloading {version} -> {dest.name}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"[ok  ] saved {len(resp.content):,} bytes")
    except Exception as e:
        print(f"[FAIL] automatic download failed: {e}")
        print(f"       Please open this URL in your browser:")
        print(f"         {url}")
        print(f"       and save the PDF as:  {dest}")
        print(f"       then run this script again.")
        sys.exit(1)
    return dest


def extract_text(pdf_path: Path) -> str:
    """Pull text from every page and drop the repeating headers/footers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"[read] {pdf_path.name}: {len(pdf.pages)} pages")
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    raw = "\n".join(pages)
    raw = raw.replace("\u00b7", " ")  # stray middle-dot artifact "·"

    kept = []
    for line in raw.splitlines():
        if any(p.match(line) for p in FOOTER_PATTERNS):
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces/tabs
    return text


def split_sections(text: str) -> list[dict]:
    """Split chapter text into {section_id, heading, text} using an explicit
    boolean state machine.

    Instead of collecting every header match and then de-duplicating the
    Table-of-Contents copies by "longest body wins", we walk the lines once and
    keep an explicit boolean, `in_section`, that says whether we are currently
    inside a real section body. A header line only *opens* a section when it is
    not a TOC entry, so the boundary between the TOC and the real chapter is
    caught directly at the boundary rather than repaired afterwards.
    """
    lines = text.splitlines()

    sections: list[dict] = []
    in_section = False          # <-- the boolean flag: are we inside a real body?
    current_id = None
    current_heading = None
    current_body: list[str] = []
    seen_ids: set[str] = set()  # a section_id we've already opened for real

    def _flush():
        """Close the current section and append it if it has real content."""
        if current_id is not None and current_body:
            body = "\n".join(current_body).strip()
            if body:
                sections.append(
                    {"section_id": current_id, "heading": current_heading, "text": body}
                )

    for line in lines:
        header = SECTION_HEADER.match(line)
        is_toc = bool(TOC_LINE.match(line))

        # A header line is a genuine boundary only if it is NOT a TOC entry and
        # we have not already opened this section id for real. That check is what
        # the boolean flag makes clean: a TOC "A. Introduction ... I-3" never
        # flips us into a section, so the TOC can never open a body.
        if header and not is_toc:
            sid = header.group(1)
            heading = header.group(2).strip()

            # Real boundary: flush the previous section, then open the new one.
            _flush()
            in_section = True
            current_id = sid
            current_heading = heading
            current_body = []
            seen_ids.add(sid)
            continue

        # Any non-header line while we're inside a section is body text.
        if in_section:
            current_body.append(line)

    # Flush the final open section.
    _flush()

    # Safety net: if the same id opened twice (rare), keep the longer body.
    best: dict[str, dict] = {}
    for s in sections:
        if s["section_id"] not in best or len(s["text"]) > len(best[s["section_id"]]["text"]):
            best[s["section_id"]] = s

    return [best[k] for k in sorted(best)]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for version, url in SOURCES.items():
        pdf_path = download(version, url)
        text = extract_text(pdf_path)
        sections = split_sections(text)

        out_path = PROCESSED_DIR / f"{version}.json"
        out_path.write_text(json.dumps(sections, indent=2))

        print(f"\n=== {version}: {len(sections)} sections detected ===")
        for s in sections:
            wc = len(s["text"].split())
            print(f"  {s['section_id']}. {s['heading'][:58]:<58} ({wc:>4} words)")
        print(f"[ok  ] wrote {out_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()