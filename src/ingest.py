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
# Chapter 1 ("General Correct Coding Policies") of the NCCI Policy Manual.
# We start with the 2025 edition only and verify parsing before adding the 2026
# edition for the diff feature in the next step.
SOURCES = {
    "2024": "https://www.cms.gov/files/document/medicare-ncci-policy-manual-2024-chapter-1.pdf",
    "2025": "https://www.cms.gov/files/document/01-chapter1-ncci-medicare-policy-manual-2025finalcleanpdf.pdf",
}

# Polite browser-like header; some .gov hosts reject the default requests UA.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CotivitiPOC/1.0; research)"}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# A real section header is a line like:
#   "B. Coding Based on Standards of Medical/Surgical Practice"
# a single capital letter, a period, then a Title-cased heading.
# NOTE: the period "." is deliberately NOT in the heading character class, so a
# Table-of-Contents line ("A. Introduction.......... I-3") cannot match here.
# (single-quoted raw string so a literal " can sit in the class; \x27 is the apostrophe)
# \.\s*  -> the period may be followed by zero spaces (CMS's 2024 PDF drops the space
# in "D.Evaluation"), one space (the usual case), or several.
SECTION_HEADER = re.compile(r'^\s*([A-Z])\.\s*([A-Z][A-Za-z0-9 ,/&()\x27’"“”-]{3,90})\s*$')

# Lines to drop entirely: the per-page "Revision Date (Medicare): ..." footer and
# the bare page-number lines like "I-3".
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
    """Split chapter text into {section_id, heading, text}, TOC-safe."""
    lines = text.splitlines()

    # 1) Find every line that looks like a section header.
    matches = []  # (line_index, section_id, heading)
    for i, line in enumerate(lines):
        m = SECTION_HEADER.match(line)
        if m:
            matches.append((i, m.group(1), m.group(2).strip()))

    # 2) For each header, the body is everything up to the next header.
    candidates = []
    for idx, (line_i, sid, heading) in enumerate(matches):
        start = line_i + 1
        end = matches[idx + 1][0] if idx + 1 < len(matches) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        candidates.append({"section_id": sid, "heading": heading, "text": body})

    # 3) The Table of Contents yields near-empty duplicates of each letter.
    #    Keep, per section_id, the candidate with the most text (the real body).
    best: dict[str, dict] = {}
    for c in candidates:
        if c["section_id"] not in best or len(c["text"]) > len(best[c["section_id"]]["text"]):
            best[c["section_id"]] = c

    # 4) Return sections in order A, B, C, ...
    return [best[k] for k in sorted(best)]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for version, url in SOURCES.items():
        pdf_path = download(version, url)
        text = extract_text(pdf_path)
        sections = split_sections(text)

        out_path = PROCESSED_DIR / f"{version}.json"
        out_path.write_text(json.dumps(sections, indent=2))

        # --- Verification summary (paste this back to confirm parsing) -------
        print(f"\n=== {version}: {len(sections)} sections detected ===")
        for s in sections:
            wc = len(s["text"].split())
            print(f"  {s['section_id']}. {s['heading'][:58]:<58} ({wc:>4} words)")
        print(f"[ok  ] wrote {out_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
