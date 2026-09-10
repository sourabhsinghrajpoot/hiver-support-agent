"""Intent classification for inbound customer messages."""
import json
from src.llm_client import complete

INTENTS = [
    "order_status", "delivery_issue", "refund_request", "billing_dispute",
    "account_access", "product_defect", "cancellation", "general_complaint",
]

SYSTEM_PROMPT = f"""TASK=CLASSIFY
You are an intent classifier for @AmazonHelp customer support tweets.
Classify the customer's message into exactly one of these intents:
{", ".join(INTENTS)}

Respond ONLY with JSON: {{"intent": "<one of the above>", "confidence": <0-1 float>}}
"""


def classify(text: str) -> dict:
    raw = complete(SYSTEM_PROMPT, text)
    try:
        result = json.loads(raw)
        if result["intent"] not in INTENTS:
            result["intent"] = "general_complaint"
            result["confidence"] = 0.3
        return result
    except (json.JSONDecodeError, KeyError):
        return {"intent": "general_complaint", "confidence": 0.2}
