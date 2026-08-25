"""
evaluate.py  --  labeled evaluation of the policy-to-rules agent.

Runs the agent on a section and scores its output against hand-labeled gold
provisions (eval/labeled_provisions.json):

  - Faithfulness: fraction of extracted rules whose source_quote is actually
                  grounded in the cited section (catches hallucinated citations).
  - Recall:       fraction of known gold provisions captured by some rule.

It also prints a failure analysis — which provisions were missed and which rules
were flagged as ungrounded — which is the honest, scientific core of the report.

Run:
    python src/evaluate.py
"""

import json
from pathlib import Path

from policy_agent import run_agent, parse_rules, verify_grounding

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "eval" / "labeled_provisions.json"


def _rule_text(rule: dict) -> str:
    """Flatten a rule's searchable fields to lowercase text."""
    return " ".join(
        str(rule.get(k, "")) for k in ("description", "condition", "source_quote")
    ).lower()


def covered_by(gold_item: dict, rules: list[dict]) -> dict | None:
    """Return the first extracted rule that contains ALL of the gold provision's
    match terms, or None if the provision was missed."""
    terms = [t.lower() for t in gold_item["terms"]]
    for r in rules:
        text = _rule_text(r)
        if all(term in text for term in terms):
            return r
    return None


def evaluate(rules: list[dict], version: str, gold: list[dict]) -> dict:
    """Score a set of extracted rules for faithfulness and recall."""
    grounded, flagged = 0, []
    for r in rules:
        g = verify_grounding(version, r.get("source_section", ""), r.get("source_quote", ""))
        if g["grounded"]:
            grounded += 1
        else:
            flagged.append((r, g))

    hits, misses = [], []
    for gi in gold:
        (hits if covered_by(gi, rules) else misses).append(gi)

    n = len(rules)
    return {
        "n_rules": n,
        "grounded": grounded,
        "faithfulness": grounded / n if n else 0.0,
        "recall": len(hits) / len(gold) if gold else 0.0,
        "hits": hits,
        "misses": misses,
        "flagged": flagged,
    }


def main():
    spec = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    version, section, gold = spec["version"], spec["section"], spec["gold"]

    task = (
        f"From the {version} NCCI Policy Manual, fetch Section {section} and "
        f"extract up to 8 of the most important, concrete coding rules as specified."
    )
    print(f"Evaluating Section {section} ({version}) against {len(gold)} labeled provisions...\n")

    rules = parse_rules(run_agent(task))
    res = evaluate(rules, version, gold)

    print(f"Extracted rules:          {res['n_rules']}")
    print(f"Faithfulness (grounded):  {res['grounded']}/{res['n_rules']}  ({res['faithfulness']:.0%})")
    print(f"Recall (gold captured):   {len(res['hits'])}/{len(gold)}  ({res['recall']:.0%})")

    if res["misses"]:
        print("\nMISSED provisions (recall failures):")
        for gi in res["misses"]:
            print(f"  - [{gi['id']}] {gi['concept']}")

    if res["flagged"]:
        print("\nUNGROUNDED rules (faithfulness flags):")
        for r, g in res["flagged"]:
            print(f"  - {r.get('rule_id')}: \"{r.get('source_quote')}\"  (score {g['score']}, {g['method']})")

    if not res["misses"] and not res["flagged"]:
        print("\nNo misses and no ungrounded rules.")


if __name__ == "__main__":
    main()
