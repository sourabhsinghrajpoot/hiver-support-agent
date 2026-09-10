"""Decide auto-handle vs escalate-to-human, with a stated reason.

Rules are intentionally simple and legible (see decision_log.md for why a
rule layer sits on top of the LLM rather than asking the LLM to also decide
escalation): escalation is a safety-critical decision, and a rule the team
can read and edit in one file is worth more here than a marginally smarter
but opaque model call.
"""

SENSITIVE_KEYWORDS = [
    "lawyer", "legal action", "sue", "fraud", "unauthorized", "scam",
    "injury", "injured", "allergic", "hospital", "police", "unsafe",
]

# Intents where even a fluent, grounded reply shouldn't go out without a human
# (account security / money movement — wrong answers here are costly)
HIGH_STAKES_INTENTS = {"account_access", "billing_dispute"}

CONFIDENCE_FLOOR = 0.55
RETRIEVAL_SIM_FLOOR = 0.15


def decide(customer_message: str, intent: str, intent_confidence: float,
           top_similarity: float) -> dict:
    text_l = customer_message.lower()

    if any(kw in text_l for kw in SENSITIVE_KEYWORDS):
        return {"escalate": True, "reason": "sensitive keyword detected (legal/fraud/safety) — always human-reviewed"}

    if intent in HIGH_STAKES_INTENTS:
        return {"escalate": True, "reason": f"high-stakes intent '{intent}' (account security or money movement) — policy requires human handling"}

    if intent_confidence < CONFIDENCE_FLOOR:
        return {"escalate": True, "reason": f"low intent-classification confidence ({intent_confidence:.2f} < {CONFIDENCE_FLOOR})"}

    if top_similarity < RETRIEVAL_SIM_FLOOR:
        return {"escalate": True, "reason": f"no sufficiently similar historical resolution found (top similarity {top_similarity:.2f} < {RETRIEVAL_SIM_FLOOR}) — reply would be ungrounded"}

    return {"escalate": False, "reason": "confident intent match, high-similarity grounded precedent, not high-stakes or sensitive"}
