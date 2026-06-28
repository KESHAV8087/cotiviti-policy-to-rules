"""
agent.py  --  the Policy-to-Rules agent (Groq free tier).

A tool-calling loop (Llama 3.3 70B on Groq) that reads NCCI policy sections via
the deterministic tools in tools.py and extracts structured, source-grounded
coding rules. After the agent returns its rules, a deterministic grounding check
verifies every rule's source_quote actually appears in the cited section,
anything unsupported is flagged as a potential hallucination.

The whole pipeline is model-agnostic; only this file talks to the LLM, so it can
be pointed at Claude/GPT/Gemini in production by swapping the client.

Setup:
    1) pip install groq
    2) Copy .env.example to .env and paste your free Groq key:  GROQ_API_KEY=...
       (get one with no credit card at https://console.groq.com)
    3) python src/agent.py
"""

import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

import tools  # local module (src/ is on sys.path when run as python src/agent.py)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Cap the section text fed to the model so a very large section (e.g. V, ~3,700
# words) stays under the free-tier tokens-per-minute limit. Grounding still runs
# against the FULL section text, so quotes drawn from this window still verify.
MAX_TOOL_TEXT = 7000

SYSTEM = """You are a payment-integrity policy analyst for a healthcare analytics company.
Your job: read a section of the CMS NCCI Policy Manual and convert its prose into
structured, machine-checkable coding rules, the kind that drive automated claim edits.

Rules of engagement:
- Use the provided tools to fetch the actual policy text. Never rely on memory of the manual.
- Extract only rules that are explicitly supported by the text. Do NOT invent or generalize.
- Each rule MUST include a verbatim source_quote (25 words or fewer) copied exactly from the
  section text, so the rule can be traced back to its origin.
- Prefer concrete, checkable rules (specific codes, modifiers, conditions) over vague principles.
- If the section has no substantive policy content (for example "Reserved for future use"),
  return an empty JSON array: []

When you have gathered what you need, output ONLY a JSON array (no prose, no markdown fences),
where each element has exactly these keys:
  "rule_id":        short id, e.g. "D1"
  "description":    one-sentence plain-language statement of the rule
  "condition":      the claim situation that triggers it
  "action":         one of "deny", "flag", "allow", "review"
  "source_section": the section letter the rule came from, e.g. "D"
  "source_quote":   <=25 words copied verbatim from the section text
"""

# OpenAI / Groq tool-schema format.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": "List all sections (id, heading, word_count) of a policy version.",
            "parameters": {
                "type": "object",
                "properties": {"version": {"type": "string", "enum": ["2024", "2025"]}},
                "required": ["version"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Get the full text of one section by its letter id (e.g. 'D').",
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {"type": "string", "enum": ["2024", "2025"]},
                    "section_id": {"type": "string"},
                },
                "required": ["version", "section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Keyword-search a policy version; returns the top matching sections with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {"type": "string", "enum": ["2024", "2025"]},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["version", "query"],
            },
        },
    },
]


def run_tool(name: str, args: dict):
    """Dispatch a tool call to the matching function in tools.py."""
    if name == "list_sections":
        return tools.list_sections(**args)
    if name == "get_section":
        return tools.get_section(**args)
    if name == "search_policy":
        return tools.search_policy(**args)
    return {"error": f"unknown tool: {name}"}


def _retry_after_seconds(msg: str):
    """Pull a 'try again in 12.5s' hint out of a Groq rate-limit message, if present."""
    m = re.search(r"try again in ([\d.]+)\s*s", msg, re.IGNORECASE)
    if m:
        try:
            return int(float(m.group(1))) + 1
        except ValueError:
            return None
    return None


def _create_with_retry(client: "Groq", **kwargs):
    """Call the chat endpoint with progressive backoff on free-tier rate limits.
    Respects Groq's own retry hint when present; gives up with a calm message."""
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            is_rate = any(k in low for k in ("rate", "429", "tokens per", "quota", "too many requests"))
            if not is_rate:
                raise
            wait = min(_retry_after_seconds(msg) or (8 * (attempt + 1)), 30)
            print(f"  [free-tier rate limit; waiting {wait}s then retry {attempt + 1}/4...]")
            time.sleep(wait)
    raise RuntimeError(
        "Groq free-tier limit reached. Please wait about a minute and run again. "
        "The pipeline is model-agnostic, so a paid model (set GROQ_MODEL or swap the client) "
        "removes this limit."
    )


def run_agent(task: str, max_steps: int = 5) -> str:
    """Run the tool-calling loop until the model returns a final (tool-free) answer.

    Guards against the smaller models' tendency to loop: it dedupes repeated
    identical tool calls (returning a cached result instead of re-fetching, which
    saves tokens and avoids free-tier rate limits), and once any section has been
    fetched it nudges the model to stop calling tools and emit the JSON. These caps
    keep a single run well under the per-minute token budget."""
    client = Groq()  # reads GROQ_API_KEY from the environment
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    seen_calls: dict[str, str] = {}   # signature -> cached JSON result
    fetched_section = False

    for step in range(1, max_steps + 1):
        # Once we have fetched a section, force the model to answer (no more tools).
        resp = _create_with_retry(
            client,
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none" if fetched_section else "auto",
            temperature=0.2,
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            return msg.content or ""

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            sig = f"{tc.function.name}:{json.dumps(args, sort_keys=True)}"

            if sig in seen_calls:
                # Repeated identical call -> return the cached result, do not re-fetch.
                print(f"  [step {step}] tool: {tc.function.name}(...) [cached, deduped]")
                content = seen_calls[sig]
            else:
                print(f"  [step {step}] tool: {tc.function.name}({json.dumps(args)})")
                result = run_tool(tc.function.name, args)
                if (tc.function.name == "get_section" and isinstance(result, dict)
                        and isinstance(result.get("text"), str)
                        and len(result["text"]) > MAX_TOOL_TEXT):
                    result = {**result, "text": result["text"][:MAX_TOOL_TEXT], "_truncated": True}
                content = json.dumps(result)
                seen_calls[sig] = content
                if tc.function.name == "get_section":
                    fetched_section = True

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

    # Final attempt: we have the text, just ask for the JSON with no tools.
    try:
        resp = _create_with_retry(
            client, model=MODEL, messages=messages,
            tool_choice="none", temperature=0.2, max_tokens=2048,
        )
        return resp.choices[0].message.content or "[]"
    except Exception:
        return "[stopped: reached max_steps without a final answer]"


# --- Grounding verification (deterministic, no LLM) -------------------------
def _norm(s: str) -> str:
    # Lowercase and strip punctuation/quotes so a near-verbatim quote that differs
    # only in punctuation (e.g. quote marks around "XXX") still matches, while a
    # quote with words actually dropped from the middle still fails the substring test.
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.;:])\s+", text)


def verify_grounding(version: str, source_section: str, quote: str) -> dict:
    """Is `quote` actually present in the cited section? Exact normalized
    substring -> grounded (1.0). Otherwise best sentence similarity; grounded
    if >= 0.85. Flags fabricated/unsupported quotes."""
    sec = tools.get_section(version, source_section or "")
    nq = _norm(quote or "")
    if not sec:
        return {"grounded": False, "score": 0.0, "method": "no-section"}
    if not nq:
        return {"grounded": False, "score": 0.0, "method": "empty-quote"}

    nt = _norm(sec["text"])
    if nq in nt:
        return {"grounded": True, "score": 1.0, "method": "exact"}

    best = 0.0
    for sent in _sentences(sec["text"]):
        best = max(best, SequenceMatcher(None, nq, _norm(sent)).ratio())
    return {"grounded": best >= 0.85, "score": round(best, 3), "method": "fuzzy"}


# --- Robust JSON parsing of the model's final answer ------------------------
def _extract_array(text: str):
    """Return the first balanced [...] substring, scanning past prose and
    respecting strings/escapes. None if no balanced array is present."""
    start = text.find("[")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ], a common LLM JSON slip."""
    return re.sub(r",(\s*[}\]])", r"\1", s)


def parse_rules(raw: str) -> list[dict]:
    """Parse the agent's final answer into a list of rule dicts, tolerating
    code fences, stray prose, trailing commas, or an object-wrapped array.
    Returns [] for an empty/"no rules" answer instead of raising."""
    if not raw or not raw.strip():
        return []
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()

    if "[stopped:" in text:
        raise ValueError("the agent did not converge in time; try again or pick a smaller section")

    def _try(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    data = _try(text)
    if data is None:
        arr = _extract_array(text)
        if arr is not None:
            data = _try(arr)
            if data is None:
                data = _try(_strip_trailing_commas(arr))

    if data is None:
        # No parseable JSON. If the model plainly said there is nothing, treat as empty.
        if re.search(r"\bno (coding )?rules\b|\bempty\b|\breserved\b", text, re.IGNORECASE):
            return []
        raise ValueError("no JSON array found in the model output")

    # If wrapped in an object like {"rules": [...]}, dig out the first list value.
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ValueError("parsed JSON is not a list of rules")
    return [r for r in data if isinstance(r, dict)]


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Put it in .env, free key at https://console.groq.com")

    version, section = "2025", "D"
    task = (
        f"From the {version} NCCI Policy Manual, fetch Section {section} "
        f"(Evaluation & Management services) and extract up to 6 of the most "
        f"important, concrete coding rules as specified."
    )

    print(f"MODEL: {MODEL}")
    print(f"TASK:  {task}\n")
    raw = run_agent(task)

    print("\n--- extracted rules + grounding check ---")
    try:
        rules = parse_rules(raw)
    except Exception as e:
        print(f"[could not parse JSON: {e}]\nRaw output:\n{raw}")
        sys.exit(1)

    if not rules:
        print("No rules extracted (section may be reserved/empty).")
        sys.exit(0)

    grounded = 0
    for r in rules:
        g = verify_grounding(version, r.get("source_section", ""), r.get("source_quote", ""))
        grounded += int(g["grounded"])
        mark = "OK  " if g["grounded"] else "FLAG"
        print(f"\n[{mark}] {r.get('rule_id')}: {r.get('description')}")
        print(f"       action={r.get('action')}  grounding={g['score']} ({g['method']})")
        print(f"       quote: \"{r.get('source_quote')}\"")

    n = len(rules)
    print(f"\nFaithfulness: {grounded}/{n} rules grounded ({(grounded / n if n else 0):.0%}).")
