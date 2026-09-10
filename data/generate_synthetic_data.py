"""
Generates a synthetic-but-structurally-identical stand-in for the real
`thoughtvector/customer-support-on-twitter` dataset, filtered to one brand.

Why synthetic: this sandbox cannot reach kaggle.com (see README_DATA.md).
Schema, noise profile, and thread structure mirror the real twcs.csv so that
every downstream module (src/, eval/) is a drop-in replacement once you have
real data.

Usage:
    python generate_synthetic_data.py --n 1500 --seed 42 --out raw_tweets.csv
"""
import argparse
import csv
import random

BRAND = "AmazonHelp"

# ---------------------------------------------------------------------------
# Intent taxonomy (see REPORT.md section "Problem framing" for why these 8
# and not, say, Banking77's 77 fine-grained classes).
# ---------------------------------------------------------------------------
INTENTS = [
    "order_status",
    "delivery_issue",
    "refund_request",
    "billing_dispute",
    "account_access",
    "product_defect",
    "cancellation",
    "general_complaint",
]

# Templates: (customer message templates, brand resolution templates, base rate)
TEMPLATES = {
    "order_status": {
        "weight": 0.20,
        "customer": [
            "hey @{brand} where is my order #{oid}?? it's been {days} days",
            "@{brand} any update on order {oid}, tracking hasn't moved since {days} days ago",
            "still waiting on order {oid} from @{brand}, no tracking info at all",
            "@{brand} placed order {oid} last week, still says 'preparing for shipment' wtf",
        ],
        "brand": [
            "Hi, sorry for the wait! I can see order {oid} is currently in transit and should arrive within 2 days. Tracking: bit.ly/track{oid} -DS",
            "So sorry about this delay. I've looked into order {oid} and it's showing movement as of today, expected delivery tomorrow. -MK",
            "I hear you, that's frustrating. Order {oid} left our facility yesterday and tracking will update within 24h. -RJ",
        ],
        "escalate_rate": 0.10,
    },
    "delivery_issue": {
        "weight": 0.20,
        "customer": [
            "@{brand} my package for order {oid} arrived DAMAGED, box was crushed",
            "never received order {oid}, @{brand} tracking says delivered but nothing here",
            "@{brand} courier left order {oid} at the wrong address, this is the 2nd time",
            "package {oid} arrived open and half the items missing @{brand}",
        ],
        "brand": [
            "I'm really sorry to hear that. I've filed a replacement for order {oid}, it'll ship today at no extra cost. -DS",
            "That's not okay, apologies. I've marked order {oid} as not-received and a reshipment is on the way. -MK",
            "So sorry for the trouble. I've escalated order {oid} to our carrier investigation team, you'll hear back in 24-48h. -RJ",
        ],
        "escalate_rate": 0.25,
    },
    "refund_request": {
        "weight": 0.15,
        "customer": [
            "@{brand} I returned order {oid} 2 weeks ago and still no refund",
            "requesting a refund for order {oid}, item was not as described",
            "@{brand} refund for {oid} is taking forever, when will i see the money back",
        ],
        "brand": [
            "Apologies for the delay. I've checked and your refund for order {oid} was processed, it can take 3-5 business days to reflect. -MK",
            "I've gone ahead and issued the refund for order {oid} directly, you should see it within 5 business days. -RJ",
            "Sorry about that! I can see the return was received; refund for {oid} is being processed now. -DS",
        ],
        "escalate_rate": 0.15,
    },
    "billing_dispute": {
        "weight": 0.10,
        "customer": [
            "@{brand} I was charged TWICE for order {oid}, unauthorized charge on my card",
            "why does my statement show two charges for order {oid}?? @{brand}",
            "@{brand} this is fraud, I never ordered {oid} but got charged $89.99",
        ],
        "brand": [
            "That's concerning, I'm sorry. Please DM us your order details so we can investigate the duplicate charge on {oid}. -MK",
            "I understand the concern - for account security I can't verify charges here, please DM us so we can look into order {oid}. -RJ",
        ],
        "escalate_rate": 0.70,
    },
    "account_access": {
        "weight": 0.10,
        "customer": [
            "@{brand} locked out of my account, reset link isn't working",
            "@{brand} someone else logged into my account, I didn't authorize this",
            "can't sign in to my account, says 'suspicious activity' @{brand}",
        ],
        "brand": [
            "Sorry for the trouble! For account security, please DM us so we can help verify and restore access. -DS",
            "That sounds concerning. Please send us a DM with your registered email so we can secure your account right away. -MK",
        ],
        "escalate_rate": 0.80,
    },
    "product_defect": {
        "weight": 0.10,
        "customer": [
            "@{brand} item from order {oid} stopped working after 2 days, defective",
            "@{brand} received the wrong item for order {oid}, ordered a charger got headphones",
            "product from {oid} is clearly used/opened, not new as advertised @{brand}",
        ],
        "brand": [
            "I'm sorry about that! I've started a replacement for order {oid}, no need to wait for the return. -RJ",
            "Apologies for the mix-up. I've flagged order {oid} for a correct replacement, it'll ship today. -DS",
        ],
        "escalate_rate": 0.15,
    },
    "cancellation": {
        "weight": 0.08,
        "customer": [
            "@{brand} need to cancel order {oid} asap, ordered by mistake",
            "how do I cancel order {oid}? placed it 10 mins ago @{brand}",
        ],
        "brand": [
            "I've cancelled order {oid} for you, you won't be charged. -MK",
            "Order {oid} has been cancelled successfully, refund (if charged) will process in 3-5 days. -DS",
        ],
        "escalate_rate": 0.05,
    },
    "general_complaint": {
        "weight": 0.07,
        "customer": [
            "@{brand} customer service has been terrible lately, third time I've had issues",
            "so done with @{brand}, nothing but problems this month",
            "@{brand} your app keeps crashing every time I try to check my orders",
        ],
        "brand": [
            "I'm really sorry to hear about your experience. I'd like to make this right, can you DM us more details? -RJ",
            "That's not the experience we want for you. Please DM us so we can look into this further. -MK",
        ],
        "escalate_rate": 0.30,
    },
}


def jitter_text(text: str, rng: random.Random) -> str:
    """Light noise to mimic real tweet messiness."""
    if rng.random() < 0.15:
        text = text.replace(" you ", " u ")
    if rng.random() < 0.10:
        text += " 😡"
    if rng.random() < 0.10:
        text += " #fail"
    if rng.random() < 0.08:
        text = text.lower()
    return text


def generate(n: int, seed: int):
    rng = random.Random(seed)
    intents_pool = []
    for intent, cfg in TEMPLATES.items():
        intents_pool += [intent] * int(cfg["weight"] * 1000)

    rows = []
    tweet_id = 1
    for i in range(n):
        intent = rng.choice(intents_pool)
        cfg = TEMPLATES[intent]
        oid = rng.randint(100000, 999999)
        days = rng.randint(2, 12)
        cust_template = rng.choice(cfg["customer"])
        brand_template = rng.choice(cfg["brand"])

        cust_text = cust_template.format(brand=BRAND, oid=oid, days=days)
        cust_text = jitter_text(cust_text, rng)
        brand_text = brand_template.format(oid=oid)

        should_escalate = rng.random() < cfg["escalate_rate"]

        cust_id = tweet_id
        brand_id = tweet_id + 1
        rows.append({
            "tweet_id": cust_id,
            "author_id": f"cust{rng.randint(10000,99999)}",
            "inbound": True,
            "created_at": f"2019-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "text": cust_text,
            "response_tweet_id": brand_id,
            "in_response_to_tweet_id": "",
            "_true_intent": intent,          # ground truth, synthetic-only
            "_should_escalate": should_escalate,  # ground truth, synthetic-only
        })
        rows.append({
            "tweet_id": brand_id,
            "author_id": BRAND,
            "inbound": False,
            "created_at": f"2019-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "text": brand_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": cust_id,
            "_true_intent": intent,
            "_should_escalate": should_escalate,
        })
        tweet_id += 2
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="number of conversation pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="raw_tweets.csv")
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tweets ({len(rows)//2} conversation pairs) to {args.out}")


if __name__ == "__main__":
    main()
