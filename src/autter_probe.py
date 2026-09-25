"""
Temporary probe file to exercise Autter's dependency and hallucinated-import
detection. Not part of the application. Safe to delete.
"""

# Case 1: package that does not exist on PyPI at all (pure hallucination).
import quantum_policy_normalizer as qpn

# Case 2: plausible typo of a real, popular package (requests -> reqeusts).
import reqeusts

# Case 3: a real, common package used in an obviously wrong way, to see
# whether execution-based checks notice runtime breakage vs just imports.
import json


def normalize_policy(raw: str) -> dict:
    parsed = qpn.normalize(raw)          # would fail: module does not exist
    payload = reqeusts.get(raw).json()   # would fail: module does not exist
    return json.loads(payload, parsed)   # wrong signature: json.loads takes one positional arg


if __name__ == "__main__":
    print(normalize_policy("test"))
