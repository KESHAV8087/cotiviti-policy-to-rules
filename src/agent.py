"""
agent.py  --  the Policy-to-Rules agent (Groq free tier).

A tool-calling loop (Llama 3.3 70B on Groq) that reads NCCI policy sections via
the deterministic tools in tools.py and extracts structured, source-grounded
coding rules. After the agent returns its rules, a deterministic grounding check
verifies every rule's source_quote actually appears in the cited section —
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

SYSTEM = """You are a payment-integrity policy analyst for a healthcare analytics company.
Your job: read a section of the CMS NCCI Policy Manual and convert its prose into
structured, machine-checkable coding rules — the kind that drive automated claim edits.

Rules of engagement:
- Use the provided tools to fetch the actual policy text. Never rely on memory of the manual.
- Extract only rules that are explicitly supported by the text. Do NOT invent or generalize.
- Each rule MUST include a verbatim source_quote (25 words or fewer) copied exactly from the
  section text, so the rule can be traced back to its origin.
- Prefer concrete, checkable rules (specific codes, modifiers, conditions) over vague principles.

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


def _create_with_retry(client: "Groq", **kwargs):
    """Call the chat endpoint, backing off briefly on free-tier rate limits."""
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            text = str(e).lower()
            if "rate" in text or "429" in text or "tokens per" in text:
                wait = 20
                print(f"  [free-tier rate limit hit; waiting {wait}s and retrying...]")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Repeated rate limits — wait a minute and run again.")


def run_agent(task: str, max_steps: int = 8) -> str:
    """Run the tool-calling loop until the model returns a final (tool-free) answer."""
    client = Groq()  # reads GROQ_API_KEY from the environment
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        resp = _create_with_retry(
            client,
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        # Record the assistant turn (must carry tool_calls when present).
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

        # Execute each requested tool and feed results back.
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  [step {step}] tool: {tc.function.name}({json.dumps(args)})")
            result = run_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

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


def parse_rules(raw: str) -> list[dict]:
    """Parse the agent's final answer into rule dicts, tolerating code fences
    or stray prose around the JSON array."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j != -1 and j > i:
            return json.loads(text[i:j + 1])
        raise


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Put it in .env — free key at https://console.groq.com")

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
