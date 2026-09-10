"""
LLM-as-judge rubric for reply quality.

Rubric (1-5, single overall score — see decision_log.md for why one score
and not four sub-scores: with only 184 golden examples and a mock judge in
this environment, four correlated sub-scores would add complexity without
adding signal; a single well-specified rubric is more reliable to hand-check):

  5 - Grounded in a plausible resolution, correct intent handling, concrete
      next step, right tone (brief, apologetic where warranted).
  4 - Mostly right, minor tone or specificity issues.
  3 - Generic but not wrong ("we'll look into it") — safe but low value.
  2 - Off-topic or ignores something important the customer said.
  1 - Wrong, unsafe, or invents policy/facts not supported by context.
"""
import json
from src.llm_client import complete

SYSTEM_PROMPT = """TASK=JUDGE
You are grading a customer support agent's draft reply for @AmazonHelp.
Score the DRAFT_REPLY from 1-5 using this rubric:
5 = grounded, correct, concrete next step, right tone
4 = mostly right, minor issues
3 = generic but safe, low value
2 = off-topic or misses something important
1 = wrong, unsafe, or invents unsupported facts
Respond ONLY with JSON: {"score": <1-5 int>, "rationale": "<one sentence>"}
"""


def judge_reply(customer_message: str, draft_reply: str) -> dict:
    user_prompt = f"CUSTOMER_MESSAGE:\n{customer_message}\nDRAFT_REPLY:\n{draft_reply}\n"
    raw = complete(SYSTEM_PROMPT, user_prompt, max_tokens=100)
    try:
        result = json.loads(raw)
        result["score"] = int(result["score"])
        return result
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 3, "rationale": "judge output unparseable, defaulted to 3"}
