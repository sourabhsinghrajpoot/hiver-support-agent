"""Load raw tweet CSV and reconstruct (customer_message, brand_resolution) pairs."""
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct threads: each inbound tweet joined to the brand tweet that
    resolved it (via response_tweet_id / in_response_to_tweet_id).

    Real-data note: in the actual Kaggle CSV, `inbound` is a string
    'True'/'False' and threads can be >2 tweets deep (customer replies again,
    brand replies again). This function keeps only the FIRST customer message
    and the brand's FIRST reply for simplicity — see decision_log.md for why
    (multi-turn threads are out of scope for v1).
    """
    df = df.copy()
    if df["inbound"].dtype == object:
        df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    inbound = df[df["inbound"]].set_index("tweet_id")
    outbound = df[~df["inbound"]].set_index("tweet_id")

    records = []
    for tid, row in inbound.iterrows():
        resp_id = row.get("response_tweet_id")
        if pd.isna(resp_id) or resp_id == "":
            continue
        resp_id = int(resp_id)
        if resp_id not in outbound.index:
            continue
        brand_row = outbound.loc[resp_id]
        records.append({
            "tweet_id": tid,
            "customer_text": row["text"],
            "brand_reply": brand_row["text"],
            "true_intent": row.get("_true_intent"),
            "should_escalate": row.get("_should_escalate"),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    df = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/raw_tweets.csv")
    pairs = build_pairs(df)
    print(f"Reconstructed {len(pairs)} customer<->brand pairs")
    print(pairs.head(3).to_string())
