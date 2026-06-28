"""
tools.py  --  deterministic tools the policy agent calls.

No LLM and no network here: these are pure-Python data-access and comparison
functions over the processed NCCI policy sections produced by ingest.py.
The agent (agent.py, next step) decides which of these to call and when.

Run a self-demo against the real parsed data:
    python src/tools.py
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

_CACHE: dict[str, list[dict]] = {}


def load_policy(version: str) -> list[dict]:
    """Load and cache the processed sections for a version ('2024' or '2025')."""
    if version not in _CACHE:
        path = PROCESSED_DIR / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python src/ingest.py")
        _CACHE[version] = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE[version]


def list_sections(version: str) -> list[dict]:
    """Return [{section_id, heading, word_count}] for a version."""
    return [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "word_count": len(s["text"].split()),
        }
        for s in load_policy(version)
    ]


def get_section(version: str, section_id: str) -> dict | None:
    """Return the full {section_id, heading, text} for one section, or None."""
    target = section_id.strip().upper()
    for s in load_policy(version):
        if s["section_id"].upper() == target:
            return s
    return None


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _snippet(text: str, q_terms: set[str], width: int = 320) -> str:
    """A short window of text around the first query-term hit."""
    low = text.lower()
    pos = -1
    for t in q_terms:
        i = low.find(t)
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        return text[:width].strip()
    start = max(0, pos - width // 3)
    end = start + width
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def search_policy(version: str, query: str, top_k: int = 3) -> list[dict]:
    """Keyword retrieval: score each section by how many query terms it contains
    (heading hits weighted 3x). Returns the top_k scoring sections, each with a
    short snippet around the first hit.

    Deliberately simple — see the report's 'limitations': a production system
    would use semantic embeddings rather than keyword overlap.
    """
    q_terms = set(_tokens(query))
    if not q_terms:
        return []

    scored = []
    for s in load_policy(version):
        body = s["text"].lower()
        head = s["heading"].lower()
        score = sum(body.count(t) for t in q_terms) + 3 * sum(head.count(t) for t in q_terms)
        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "section_id": s["section_id"],
            "heading": s["heading"],
            "score": score,
            "snippet": _snippet(s["text"], q_terms),
        }
        for score, s in scored[:top_k]
    ]


def diff_policies(v_old: str = "2024", v_new: str = "2025") -> dict:
    """Compare two versions section-by-section.

    Returns a structured report: sections added/removed between versions,
    sections whose heading changed, and a text-change score for shared sections
    (0.0 = identical, 1.0 = completely different), via difflib similarity.
    """
    old = {s["section_id"]: s for s in load_policy(v_old)}
    new = {s["section_id"]: s for s in load_policy(v_new)}

    report = {"old": v_old, "new": v_new, "added": [], "removed": [], "changed": [], "unchanged": []}

    for sid in sorted(set(old) | set(new)):
        if sid in new and sid not in old:
            report["added"].append({"section_id": sid, "heading": new[sid]["heading"]})
        elif sid in old and sid not in new:
            report["removed"].append({"section_id": sid, "heading": old[sid]["heading"]})
        else:
            o, n = old[sid], new[sid]
            ratio = SequenceMatcher(None, o["text"], n["text"]).ratio()
            entry = {
                "section_id": sid,
                "heading_old": o["heading"],
                "heading_new": n["heading"],
                "heading_changed": o["heading"] != n["heading"],
                "text_change": round(1 - ratio, 3),  # 0 = identical, 1 = fully changed
            }
            # Treat tiny ratios as noise; flag a real change otherwise.
            if entry["heading_changed"] or entry["text_change"] > 0.02:
                report["changed"].append(entry)
            else:
                report["unchanged"].append(entry)

    return report


# --- Self-demo (run: python src/tools.py) ----------------------------------
if __name__ == "__main__":
    print(f"2025 has {len(list_sections('2025'))} sections.\n")

    print("search_policy('2025', 'modifier 59 bypass edit'):")
    for r in search_policy("2025", "modifier 59 bypass edit"):
        print(f"  [{r['score']:>3}] {r['section_id']}. {r['heading']}")

    print("\ndiff_policies('2024', '2025'):")
    rep = diff_policies()
    for e in rep["changed"]:
        kind = "heading+text" if e["heading_changed"] else "text"
        print(f"  {e['section_id']}: change={e['text_change']:.2f} ({kind})")
        if e["heading_changed"]:
            print(f"       '{e['heading_old']}'  ->  '{e['heading_new']}'")
    print(f"  added={[a['section_id'] for a in rep['added']]}  "
          f"removed={[r['section_id'] for r in rep['removed']]}  "
          f"unchanged={len(rep['unchanged'])}")
