"""End-to-end: customer message -> intent, draft reply, escalation decision."""
from src.intents import classify
from src.reply_gen import generate_reply
from src.escalation import decide


class SupportAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def handle(self, customer_message: str) -> dict:
        clf = classify(customer_message)
        intent, confidence = clf["intent"], clf["confidence"]

        retrieved = self.retriever.retrieve(customer_message, k=3, intent=intent)
        top_sim = retrieved[0]["similarity"] if retrieved else 0.0

        esc = decide(customer_message, intent, confidence, top_sim)

        reply = None
        if not esc["escalate"]:
            reply = generate_reply(customer_message, retrieved)

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "top_retrieval_similarity": top_sim,
            "escalate": esc["escalate"],
            "escalation_reason": esc["reason"],
            "draft_reply": reply,
        }
