"""
tools.py  --  deterministic tools the policy agent calls.

No LLM here: these are data-access and comparison functions over the processed
NCCI policy sections produced by ingest.py. The agent (agent.py) decides which
to call and when.

Retrieval comes in two flavours:
  * search_policy()          -- keyword / lexical overlap (the original baseline)
  * search_policy_semantic() -- vector embeddings + cosine similarity (the upgrade)

Keeping both on purpose: the honest way to adopt embeddings is to measure them
against the keyword baseline on this narrow, jargon-heavy corpus rather than
assume they win. search_policy_hybrid() blends the two.

Run a self-demo against the real parsed data:
    python src/tools.py
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EMBED_DIR = ROOT / "data" / "embeddings"

_CACHE: dict[str, list[dict]] = {}

# Lazily-built embedding state, per version: the model, the section vectors, and
# the section metadata they line up with. Kept module-level so we embed once.
_EMBED_CACHE: dict[str, dict] = {}
_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for a POC corpus


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


# --- Keyword retrieval (original baseline) ---------------------------------
def search_policy(version: str, query: str, top_k: int = 3) -> list[dict]:
    """Keyword retrieval: score each section by how many query terms it contains
    (heading hits weighted 3x). Returns the top_k scoring sections, each with a
    short snippet around the first hit.

    Simple by design; the semantic version below is the upgrade path.
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


# --- Semantic retrieval (embeddings + cosine similarity) -------------------
def _get_model():
    """Load the sentence-embedding model once (lazy import so the keyword path
    has no heavy dependency)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Semantic search needs sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from e
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _build_embeddings(version: str) -> dict:
    """Embed every section once (heading + body) and cache the vectors.

    Persists to data/embeddings/<version>.npz so we don't re-embed on every run.
    Returns {"vectors": ndarray (n, d), "sections": list[dict]}.
    """
    if version in _EMBED_CACHE:
        return _EMBED_CACHE[version]

    import numpy as np

    sections = load_policy(version)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBED_DIR / f"{version}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        vectors = data["vectors"]
        # Guard: if the corpus changed size, rebuild rather than use stale vectors.
        if vectors.shape[0] == len(sections):
            _EMBED_CACHE[version] = {"vectors": vectors, "sections": sections}
            return _EMBED_CACHE[version]

    model = _get_model()
    # Embed heading + body together so the section topic is represented.
    texts = [f"{s['heading']}. {s['text']}" for s in sections]
    # normalize_embeddings=True makes a plain dot product equal cosine similarity.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    np.savez(cache_path, vectors=vectors)
    _EMBED_CACHE[version] = {"vectors": vectors, "sections": sections}
    return _EMBED_CACHE[version]


def search_policy_semantic(version: str, query: str, top_k: int = 3) -> list[dict]:
    """Semantic retrieval: embed the query and rank sections by cosine similarity
    to the query vector. Matches on meaning, so it catches paraphrases and
    synonyms that keyword overlap misses.

    Cosine similarity here = dot product, because vectors are normalized. It
    compares the *direction* of the query vector against each section vector.
    """
    import numpy as np

    query = (query or "").strip()
    if not query:
        return []

    store = _build_embeddings(version)
    vectors, sections = store["vectors"], store["sections"]

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")[0]

    # Normalized vectors -> dot product is the cosine similarity (direction match).
    sims = vectors @ q_vec  # shape (n,)

    # top_k highest-similarity sections.
    order = np.argsort(-sims)[:top_k]
    q_terms = set(_tokens(query))
    results = []
    for idx in order:
        s = sections[int(idx)]
        results.append(
            {
                "section_id": s["section_id"],
                "heading": s["heading"],
                "score": round(float(sims[int(idx)]), 4),  # cosine similarity, 0..1
                "snippet": _snippet(s["text"], q_terms),
            }
        )
    return results


# --- Hybrid retrieval (blend keyword + semantic) ---------------------------
def search_policy_hybrid(version: str, query: str, top_k: int = 3,
                         alpha: float = 0.5) -> list[dict]:
    """Combine keyword and semantic scores. alpha weights semantic vs keyword
    (1.0 = pure semantic, 0.0 = pure keyword). Scores are min-max normalized
    within each method before blending so they are comparable.
    """
    import numpy as np

    kw = {r["section_id"]: r for r in search_policy(version, query, top_k=999)}
    sem = {r["section_id"]: r for r in search_policy_semantic(version, query, top_k=999)}

    def _norm(d, key):
        if not d:
            return {}
        vals = [r[key] for r in d.values()]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return {sid: (r[key] - lo) / span for sid, r in d.items()}

    kw_n = _norm(kw, "score")
    sem_n = _norm(sem, "score")

    ids = set(kw_n) | set(sem_n)
    blended = []
    for sid in ids:
        score = alpha * sem_n.get(sid, 0.0) + (1 - alpha) * kw_n.get(sid, 0.0)
        meta = sem.get(sid) or kw.get(sid)
        blended.append((score, meta))

    blended.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "section_id": m["section_id"],
            "heading": m["heading"],
            "score": round(float(score), 4),
            "snippet": m["snippet"],
        }
        for score, m in blended[:top_k]
    ]


def diff_policies(v_old: str = "2024", v_new: str = "2025") -> dict:
    """Compare two versions section-by-section (unchanged from baseline)."""
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
                "text_change": round(1 - ratio, 3),
            }
            if entry["heading_changed"] or entry["text_change"] > 0.02:
                report["changed"].append(entry)
            else:
                report["unchanged"].append(entry)

    return report


# --- Self-demo (run: python src/tools.py) ----------------------------------
if __name__ == "__main__":
    print(f"2025 has {len(list_sections('2025'))} sections.\n")

    q = "modifier 59 bypass edit"
    print(f"search_policy (keyword) '{q}':")
    for r in search_policy("2025", q):
        print(f"  [{r['score']:>3}] {r['section_id']}. {r['heading']}")

    print(f"\nsearch_policy_semantic (embeddings) '{q}':")
    try:
        for r in search_policy_semantic("2025", q):
            print(f"  [{r['score']:.3f}] {r['section_id']}. {r['heading']}")
    except ImportError as e:
        print(f"  (skipped) {e}")

    print(f"\nsearch_policy_hybrid '{q}':")
    try:
        for r in search_policy_hybrid("2025", q):
            print(f"  [{r['score']:.3f}] {r['section_id']}. {r['heading']}")
    except ImportError as e:
        print(f"  (skipped) {e}")

    print("\ndiff_policies('2024', '2025'):")
    rep = diff_policies()
    for e in rep["changed"]:
        kind = "heading+text" if e["heading_changed"] else "text"
        print(f"  {e['section_id']}: change={e['text_change']:.2f} ({kind})")
    print(f"  added={[a['section_id'] for a in rep['added']]}  "
          f"removed={[r['section_id'] for r in rep['removed']]}  "
          f"unchanged={len(rep['unchanged'])}")