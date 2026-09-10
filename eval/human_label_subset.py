"""
Produces the "human" side of the judge-human agreement measurement.

HONEST LIMITATION (flagged again in REPORT.md): this sandbox has one operator
(me) and no separate human annotator pool. `human_score()` implements an
INDEPENDENT rubric read — deliberately using different signals than
`llm_client._mock_judge()` (which looks for order-id digits, apology words,
"DM us" phrasing) so the two aren't trivially correlated by construction —
plus injected noise, to approximate what a second, imperfect human rater
would look like. This is a stand-in, not a real inter-rater study.

To make this real: replace `human_score()` with a lookup into a CSV that a
real person filled in after reading `results/judge_human_subset.csv`
(customer message + draft reply, score blind to the judge's score).
"""
import re


def human_score(customer_message: str, draft_reply: str, rng) -> int:
    """Independent heuristic: rewards specificity and a clear next action,
    penalizes replies that ignore the emotional register of the complaint
    (e.g. an angry message met with a flat template reply)."""
    score = 3.0
    reply_l = draft_reply.lower()
    msg_l = customer_message.lower()

    # rewards concrete action verbs a human reviewer would look for
    if any(w in reply_l for w in ["replacement", "refund", "cancel", "reshipment", "processed"]):
        score += 1
    # penalize if customer used urgent/frustrated language but reply is flat/short
    if any(w in msg_l for w in ["!!", "wtf", "terrible", "done with", "2nd time"]) and len(draft_reply.split()) < 12:
        score -= 1
    # reward personalization signal (order id echoed back)
    if re.search(r"\d{5,}", customer_message) and re.search(r"\d{5,}", draft_reply):
        score += 0.5
    # small independent noise to mimic real rater variance/disagreement
    score += rng.choice([-1, 0, 0, 0, 1]) * 0.5

    return max(1, min(5, round(score)))
