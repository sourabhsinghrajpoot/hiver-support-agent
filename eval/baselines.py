"""
Two baselines the main system must beat (mandatory per the assignment):

TRIVIAL: always predict the single most common intent in the training pairs;
         always escalate (a support team that trusts nothing is "safe" but
         useless — this baseline exists to show that number is easy to beat
         but not meaningless as a floor).

SIMPLE: TF-IDF + Logistic Regression intent classifier trained on a held-out
        split of the historical pairs, no LLM, no retrieval. Reply is a fixed
        per-intent template (no personalization). Escalation is a single
        static rule: escalate iff intent in a fixed high-stakes set.
        This is what a reasonably competent team could ship in a day without
        touching an LLM at all — the bar the "AI system" actually has to clear.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TEMPLATE_REPLIES = {
    "order_status": "Thanks for reaching out! Please allow 24-48h for tracking to update. -Support",
    "delivery_issue": "So sorry about this. Please DM your order number so we can look into it. -Support",
    "refund_request": "Refunds typically take 3-5 business days to process once initiated. -Support",
    "billing_dispute": "Please DM us your order details so we can investigate the charge. -Support",
    "account_access": "For account security, please DM us your registered email to proceed. -Support",
    "product_defect": "Sorry to hear that. Please DM your order number so we can arrange a replacement. -Support",
    "cancellation": "We've received your cancellation request and are processing it now. -Support",
    "general_complaint": "We're sorry to hear about your experience. Please DM us more details. -Support",
}
STATIC_ESCALATE_INTENTS = {"account_access", "billing_dispute"}


class TrivialBaseline:
    def __init__(self, pairs_df: pd.DataFrame):
        self.majority_intent = pairs_df["true_intent"].value_counts().idxmax()

    def predict_intent(self, text: str) -> str:
        return self.majority_intent

    def predict_escalate(self, text: str, intent: str) -> bool:
        return True  # trivial baseline always escalates

    def generate_reply(self, text: str, intent: str) -> str | None:
        return None  # trivial baseline never auto-replies


class SimpleBaseline:
    def __init__(self, pairs_df: pd.DataFrame, seed: int = 42):
        train, _ = train_test_split(pairs_df, test_size=0.2, random_state=seed, stratify=pairs_df["true_intent"])
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        X = self.vectorizer.fit_transform(train["customer_text"])
        y = train["true_intent"]
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)

    def predict_intent(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.clf.predict(X)[0]

    def predict_escalate(self, text: str, intent: str) -> bool:
        return intent in STATIC_ESCALATE_INTENTS

    def generate_reply(self, text: str, intent: str) -> str | None:
        if self.predict_escalate(text, intent):
            return None
        return TEMPLATE_REPLIES.get(intent, TEMPLATE_REPLIES["general_complaint"])
