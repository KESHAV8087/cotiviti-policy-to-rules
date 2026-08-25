"""
app.py  --  Streamlit UI for the NCCI Policy-to-Rules agent.

Three tabs map to the three capabilities in the assessment topic:
  - Summarize & extract: plain-language summary + agentic, source-grounded rules
  - Compare versions:     deterministic 2024 -> 2025 section diff
  - Evaluation:           faithfulness / recall against hand-labeled provisions

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

import tools            # noqa: E402
import policy_agent as agent           # noqa: E402  (run_agent, parse_rules, verify_grounding, MODEL, _create_with_retry)
import evaluate as ev   # noqa: E402  (evaluate, gold scoring)
from groq import Groq   # noqa: E402

# --- page setup & light styling --------------------------------------------
st.set_page_config(page_title="Policy-to-Rules", page_icon="📋", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f7f8f9; }
      .rule-card { background:#ffffff; border:1px solid #e6e8eb; border-left:4px solid #0f766e;
                   border-radius:8px; padding:14px 16px; margin-bottom:12px; }
      .rule-card-flag { border-left-color:#dc2626; }
      .rule-head { font-weight:600; color:#0f172a; margin-bottom:4px; }
      .pill { display:inline-block; font-size:12px; font-weight:600; padding:2px 9px;
              border-radius:999px; margin-right:6px; }
      .ok   { background:#dcfce7; color:#166534; }
      .flag { background:#fee2e2; color:#991b1b; }
      .act-deny  { background:#fee2e2; color:#991b1b; }
      .act-allow { background:#dcfce7; color:#166534; }
      .act-flag  { background:#fef3c7; color:#92400e; }
      .act-review{ background:#e2e8f0; color:#334155; }
      .quote { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; color:#475569;
               background:#f1f5f9; padding:6px 10px; border-radius:6px; margin-top:6px; }
      .reason { font-size:12px; color:#991b1b; margin-top:8px; font-weight:500; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NCCI Policy-to-Rules")
st.caption(
    "Agentic conversion of CMS coding policy into structured, source-grounded edits, "
    "with a hallucination guardrail. Source: CMS NCCI Policy Manual, Chapter 1 (public domain)."
)


# --- shared helpers ---------------------------------------------------------
def _is_rate_limit(e) -> bool:
    return "limit" in str(e).lower()


def _show_llm_error(e):
    """Calm yellow notice for free-tier limits; red only for real failures."""
    if _is_rate_limit(e):
        st.warning(f"⏳ {e}")
    else:
        st.error(f"Something went wrong: {e}")


def _too_soon(min_gap: float = 4.0) -> bool:
    """Light client-side cooldown so rapid clicks do not trip the free-tier limit."""
    now = time.time()
    if now - st.session_state.get("_last_llm", 0.0) < min_gap:
        return True
    st.session_state["_last_llm"] = now
    return False


def _section_picker(key: str):
    """Shared edition + section selector. Returns (version, section_id)."""
    c1, c2 = st.columns([1, 3])
    version = c1.selectbox("Edition", ["2025", "2024"], key=f"ver_{key}")
    secs = tools.list_sections(version)
    labels = {f"{s['section_id']}. {s['heading']}": s["section_id"] for s in secs}
    chosen = c2.selectbox("Section", list(labels), key=f"sec_{key}")
    return version, labels[chosen]


def _grounding_reason(g: dict) -> str:
    """Human-readable explanation for why a rule's quote was flagged."""
    if not isinstance(g, dict):
        return "quote could not be grounded in the cited section"
    method = g.get("method")
    score = g.get("score", 0.0)
    if method == "no-section":
        return "the cited section was not found in this edition"
    if method == "empty-quote":
        return "no source quote was provided"
    return f"quote not found verbatim in the cited section (closest match {score:.2f})"


def summarize_section(version: str, section_id: str) -> str:
    """One plain-language summary of a section (single, non-agentic LLM call)."""
    sec = tools.get_section(version, section_id)
    if not sec:
        return "Section not found."
    if len((sec.get("text") or "").split()) < 15:
        return (f"Section {section_id} ('{sec.get('heading', '')}') has no substantive "
                f"policy text to summarize (it reads as 'Reserved for future use' or is empty).")
    client = Groq()
    resp = agent._create_with_retry(
        client,
        model=agent.MODEL,
        temperature=0.2,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You summarize CMS coding policy for a payment-integrity "
             "analyst. 3-4 plain-language sentences. No preamble, no markdown."},
            {"role": "user", "content": f"Summarize this policy section titled "
             f"'{sec['heading']}':\n\n{sec['text'][:6000]}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def render_rules(rules: list[dict], version: str):
    """Render extracted rules as cards with action + grounding badges."""
    grounded = 0
    for r in rules:
        g = agent.verify_grounding(version, r.get("source_section", ""), r.get("source_quote", ""))
        is_grounded = g["grounded"]
        grounded += int(is_grounded)
        gb = ("ok", f"grounded {g['score']:.2f}") if is_grounded else ("flag", f"flagged {g['score']:.2f}")
        act = (r.get("action") or "review").lower()
        card_cls = "rule-card" if is_grounded else "rule-card rule-card-flag"
        reason_html = "" if is_grounded else (
            f'<div class="reason">⚠ Flagged: {_grounding_reason(g)}</div>'
        )
        rid = r.get("rule_id", "")
        desc = r.get("description", "")
        cond = r.get("condition", "")
        quote = r.get("source_quote", "")
        card_html = (
            f'<div class="{card_cls}">'
            f'<div class="rule-head">{rid} · {desc}</div>'
            f'<span class="pill act-{act}">{act}</span>'
            f'<span class="pill {gb[0]}">{gb[1]}</span>'
            f'<div style="font-size:13px;color:#475569;margin-top:6px">{cond}</div>'
            f'<div class="quote">“{quote}”</div>'
            f'{reason_html}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    n = len(rules)
    st.metric("Faithfulness", f"{grounded}/{n}", f"{(grounded / n if n else 0):.0%} grounded")


tab1, tab2, tab3 = st.tabs(["Summarize & extract", "Compare versions", "Evaluation"])

# --- Tab 1: summarize + extract grounded rules ------------------------------
with tab1:
    version, sid = _section_picker("extract")
    c1, c2 = st.columns(2)

    if c1.button("Summarize section", width="stretch"):
        if _too_soon():
            st.info("Please wait a few seconds between runs to stay within the free-tier limit.")
        else:
            with st.spinner("Summarizing…"):
                try:
                    st.session_state["summary"] = summarize_section(version, sid)
                except Exception as e:
                    st.session_state.pop("summary", None)
                    _show_llm_error(e)
    if st.session_state.get("summary"):
        st.info(st.session_state["summary"])

    if c2.button("Extract grounded rules", type="primary", width="stretch"):
        sec = tools.get_section(version, sid)
        wc = len((sec.get("text") or "").split()) if sec else 0
        if wc < 15:
            st.session_state.pop("rules", None)
            st.warning(
                f"Section {sid} has no substantive policy text to extract "
                f"(it reads as 'Reserved for future use' or is empty). "
                f"Try a content-rich section such as D, E, or C."
            )
        elif _too_soon():
            st.info("Please wait a few seconds between runs to stay within the free-tier limit.")
        else:
            with st.spinner("Agent reading the policy and extracting rules…"):
                task = (f"From the {version} NCCI Policy Manual, fetch Section {sid} and extract "
                        f"up to 8 of the most important, concrete coding rules as specified.")
                try:
                    st.session_state["rules"] = agent.parse_rules(agent.run_agent(task))
                    st.session_state["rules_version"] = version
                except Exception as e:
                    st.session_state.pop("rules", None)
                    _show_llm_error(e)

    if st.session_state.get("rules"):
        st.subheader("Extracted rules")
        render_rules(st.session_state["rules"], st.session_state.get("rules_version", version))
    elif st.session_state.get("rules") == []:
        st.info("No extractable rules were found in this section.")

# --- Tab 2: version diff (deterministic) ------------------------------------
with tab2:
    st.write("Section-level changes between editions, the signal that a policy update should "
             "trigger a rule review (text-change 0 = identical, 1 = fully rewritten).")
    rep = tools.diff_policies("2024", "2025")

    changed = rep["changed"]
    if changed:
        rows = [{
            "Section": e["section_id"],
            "Heading (2025)": e["heading_new"],
            "Heading changed": "yes" if e["heading_changed"] else "no",
            "Text change": e["text_change"],
        } for e in changed]
        st.dataframe(rows, width="stretch", hide_index=True)

        st.caption(f"Added: {[a['section_id'] for a in rep['added']] or 'none'} · "
                   f"Removed: {[r['section_id'] for r in rep['removed']] or 'none'} · "
                   f"Unchanged: {len(rep['unchanged'])}")

        ids = [e["section_id"] for e in changed]
        pick = st.selectbox("Explain a changed section in plain language", ids)
        if st.button("Explain this change"):
            if _too_soon():
                st.info("Please wait a few seconds between runs to stay within the free-tier limit.")
            else:
                o = tools.get_section("2024", pick)
                n = tools.get_section("2025", pick)
                head_o = (o or {}).get("heading", "")
                head_n = (n or {}).get("heading", "")
                text_o = (o or {}).get("text", "").strip()
                text_n = (n or {}).get("text", "").strip()
                # Describe an empty side explicitly so the model does not say "not provided".
                body_o = text_o[:3000] if text_o else "(this section had no body text in 2024)"
                body_n = text_n[:3000] if text_n else "(this section is empty in 2025; it appears to have been retired or reserved)"
                with st.spinner("Comparing the two versions…"):
                    try:
                        client = Groq()
                        resp = agent._create_with_retry(
                            client,
                            model=agent.MODEL, temperature=0.2, max_tokens=350,
                            messages=[
                                {"role": "system", "content": "You explain what changed between two "
                                 "versions of a CMS coding policy section, including when a section was "
                                 "retired or reserved. 2-4 plain sentences. No preamble."},
                                {"role": "user", "content": f"Section {pick}.\n"
                                 f"2024 heading: {head_o}\n2024 text:\n{body_o}\n\n"
                                 f"2025 heading: {head_n}\n2025 text:\n{body_n}\n\n"
                                 f"What changed between 2024 and 2025?"},
                            ],
                        )
                        st.info(resp.choices[0].message.content.strip())
                    except Exception as e:
                        _show_llm_error(e)
    else:
        st.write("No changed sections detected.")

# --- Tab 3: evaluation ------------------------------------------------------
with tab3:
    st.write("Run the labeled evaluation: faithfulness (are cited quotes real?) and "
             "recall (are known provisions captured?).")
    if st.button("Run evaluation", type="primary"):
        if _too_soon():
            st.info("Please wait a few seconds between runs to stay within the free-tier limit.")
        else:
            import json
            spec = json.loads((ROOT / "eval" / "labeled_provisions.json").read_text(encoding="utf-8"))
            with st.spinner("Running the agent and scoring against gold provisions…"):
                try:
                    task = (f"From the {spec['version']} NCCI Policy Manual, fetch Section "
                            f"{spec['section']} and extract up to 8 of the most important, "
                            f"concrete coding rules as specified.")
                    rules = agent.parse_rules(agent.run_agent(task))
                    res = ev.evaluate(rules, spec["version"], spec["gold"])
                    st.session_state["eval"] = (res, spec["gold"])
                except Exception as e:
                    st.session_state.pop("eval", None)
                    _show_llm_error(e)

    if st.session_state.get("eval"):
        res, gold = st.session_state["eval"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Rules extracted", res["n_rules"])
        m2.metric("Faithfulness", f"{res['faithfulness']:.0%}", f"{res['grounded']}/{res['n_rules']} grounded")
        m3.metric("Recall", f"{res['recall']:.0%}", f"{len(res['hits'])}/{len(gold)} provisions")
        if res["misses"]:
            st.warning("Missed provisions: " + "; ".join(g["concept"] for g in res["misses"]))
        if res["flagged"]:
            items = [f'{r.get("rule_id")}: {_grounding_reason(g)}' for r, g in res["flagged"]]
            st.error("Ungrounded rules — " + "; ".join(items))
        if not res["misses"] and not res["flagged"]:
            st.success("All extracted rules grounded and all labeled provisions captured.")
