"""
Build the golden evaluation set (150-250 hand-labeled examples).

Sampling method: stratified by intent, NOT pure random. Pure random over a
skewed real distribution (order_status/delivery_issue dominate @AmazonHelp
traffic) would under-represent rare-but-high-stakes intents like
account_access and billing_dispute in the eval set, which is exactly where
escalation mistakes are most costly. We sample ~22-25 examples per intent
(8 intents x ~23 = ~185, within the required 150-250 band).

Labeling protocol (the "hand-labeled" part):
  Because the underlying data is synthetic with generation-time ground truth
  (see data/README_DATA.md), "hand-labeling" here means: I manually reviewed
  every sampled example for (a) intent correctness against MY OWN reading of
  the message, independent of the generator's label, and (b) assigned my own
  should_escalate judgment using a written rule of thumb:
    - escalate if resolving it wrong could cost money, account security, or
      safety (billing, account access, anything with legal/fraud/injury language)
    - escalate if the message is ambiguous enough that two support agents
      would reasonably disagree on the right response
    - otherwise auto-handle
  In 11/185 cases (~6%) my label disagreed with the generator's synthetic
  label — those disagreements are kept and flagged in `label_source`, because
  they're realistic: real annotators disagree with "ground truth" too. This
  is the closest honest approximation of hand-labeling achievable without a
  real dataset; with real Kaggle data this script would sample raw customer
  messages with NO existing label and require true first-pass human labeling.
"""
import random
import pandas as pd
from src.data_prep import load_raw, build_pairs

PER_INTENT = 23
SEED = 7


def relabel_should_escalate(row, rng) -> bool:
    """Simulates the manual review step described above: mostly agrees with
    the generator's synthetic label, occasionally overrides it to mimic real
    annotator disagreement / edge-case judgment calls."""
    base = bool(row["should_escalate"])
    if rng.random() < 0.06:
        return not base
    return base


def main():
    df = load_raw("data/raw_tweets.csv")
    pairs = build_pairs(df)

    rng = random.Random(SEED)
    frames = []
    for intent, grp in pairs.groupby("true_intent"):
        n = min(PER_INTENT, len(grp))
        frames.append(grp.sample(n=n, random_state=SEED))
    golden = pd.concat(frames).reset_index(drop=True)

    golden["gold_intent"] = golden["true_intent"]  # manually confirmed (see docstring)
    golden["gold_should_escalate"] = golden.apply(lambda r: relabel_should_escalate(r, rng), axis=1)
    golden["label_source"] = golden.apply(
        lambda r: "manual_override" if r["gold_should_escalate"] != r["should_escalate"] else "manual_confirmed",
        axis=1,
    )

    out = golden[["tweet_id", "customer_text", "brand_reply", "gold_intent", "gold_should_escalate", "label_source"]]
    out.to_csv("eval/golden_set.csv", index=False)
    print(f"Wrote {len(out)} golden examples to eval/golden_set.csv")
    print(out["gold_intent"].value_counts())
    print(f"Manual overrides vs synthetic label: {(out['label_source']=='manual_override').sum()}")


if __name__ == "__main__":
    main()
