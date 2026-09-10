"""Retrieve similar historically-resolved conversations to ground reply generation.

Design choice (see decision_log.md): TF-IDF cosine similarity, not embeddings.
For short, templated brand-support tweets, lexical overlap ("damaged", "refund",
"locked out") carries almost all the signal, and TF-IDF is free, offline, and
auditable — you can see exactly why an example was retrieved. An embedding
retriever is listed under "what I'd do next."
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.pairs_df["customer_text"])

    def retrieve(self, query: str, k: int = 3, intent: str | None = None):
        pool = self.pairs_df
        matrix = self.matrix
        if intent is not None and "true_intent" in pool.columns:
            mask = pool["true_intent"] == intent
            if mask.sum() >= k:  # only restrict if enough examples exist
                pool = pool[mask]
                matrix = self.matrix[mask.values]

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            row = pool.iloc[idx]
            results.append({
                "customer_text": row["customer_text"],
                "brand_reply": row["brand_reply"],
                "similarity": float(sims[idx]),
            })
        return results
