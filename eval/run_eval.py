"""
Runs all headline metrics and writes results/metrics_summary.md + .json.

What this computes:
  1. Intent classification: accuracy + macro-F1, main system vs both baselines.
  2. Escalation decision: precision/recall/F1 on "should escalate" vs golden
     labels, main system vs both baselines. False negatives (auto-handled but
     should've escalated) reported separately — that's the costly error class.
  3. Reply quality: mean LLM-judge score (1-5) on auto-handled replies only
     (baselines and system that chose to escalate have no reply to score).
  4. Judge-human agreement: quadratic-weighted Cohen's kappa + Pearson r
     between the judge's scores and independently-produced human scores on a
     40-example subset (see human_label_subset.py for how "human" labels were
     produced in this sandbox — flagged clearly as a limitation).
"""
import json
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

from src.data_prep import load_raw, build_pairs
from src.retrieval import Retriever
from src.pipeline import SupportAgent
from eval.baselines import TrivialBaseline, SimpleBaseline
from eval.judge import judge_reply

RESULTS_DIR = "results"


def eval_intent_and_escalation(name, predict_intent_fn, predict_escalate_fn, golden):
    pred_intents, pred_escalates = [], []
    for _, row in golden.iterrows():
        intent = predict_intent_fn(row["customer_text"])
        pred_intents.append(intent)
        pred_escalates.append(predict_escalate_fn(row["customer_text"], intent))

    gold_intents = golden["gold_intent"].tolist()
    gold_escalates = golden["gold_should_escalate"].tolist()

    intent_acc = accuracy_score(gold_intents, pred_intents)
    intent_f1 = f1_score(gold_intents, pred_intents, average="macro")

    esc_precision = precision_score(gold_escalates, pred_escalates, zero_division=0)
    esc_recall = recall_score(gold_escalates, pred_escalates, zero_division=0)
    esc_f1 = f1_score(gold_escalates, pred_escalates, zero_division=0)

    # false negatives: predicted auto-handle but gold says should escalate — the costly error
    fn = sum(1 for g, p in zip(gold_escalates, pred_escalates) if g and not p)

    return {
        "system": name,
        "intent_accuracy": round(intent_acc, 3),
        "intent_macro_f1": round(intent_f1, 3),
        "escalation_precision": round(esc_precision, 3),
        "escalation_recall": round(esc_recall, 3),
        "escalation_f1": round(esc_f1, 3),
        "costly_false_negatives": fn,  # auto-handled but should have escalated
        "n": len(golden),
    }, pred_intents, pred_escalates


def score_replies(name, replies_and_msgs):
    """replies_and_msgs: list of (customer_message, draft_reply) for auto-handled cases only."""
    if not replies_and_msgs:
        return {"system": name, "mean_judge_score": None, "n_replies_scored": 0}
    scores = [judge_reply(msg, reply)["score"] for msg, reply in replies_and_msgs]
    return {
        "system": name,
        "mean_judge_score": round(sum(scores) / len(scores), 2),
        "n_replies_scored": len(scores),
    }


def judge_human_agreement(golden, agent, n=40, seed=11):
    """Run the main system on a subset, judge the replies, and independently
    'human'-score the same replies to measure judge-human agreement.
    See human_label_subset.py for how the human labels are produced here."""
    from eval.human_label_subset import human_score

    rng = random.Random(seed)
    auto_handled = []
    for _, row in golden.sample(frac=1, random_state=seed).iterrows():
        result = agent.handle(row["customer_text"])
        if not result["escalate"]:
            auto_handled.append((row["customer_text"], result["draft_reply"]))
        if len(auto_handled) >= n:
            break

    judge_scores, human_scores = [], []
    rows = []
    for msg, reply in auto_handled:
        j = judge_reply(msg, reply)["score"]
        h = human_score(msg, reply, rng)
        judge_scores.append(j)
        human_scores.append(h)
        rows.append({"customer_text": msg, "draft_reply": reply, "judge_score": j, "human_score": h})

    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/judge_human_subset.csv", index=False)

    kappa = cohen_kappa_score(judge_scores, human_scores, weights="quadratic") if len(set(judge_scores)) > 1 and len(set(human_scores)) > 1 else float("nan")
    r, _ = pearsonr(judge_scores, human_scores) if len(judge_scores) > 1 else (float("nan"), None)
    return {"n": len(rows), "quadratic_weighted_kappa": round(kappa, 3) if kappa == kappa else None,
            "pearson_r": round(r, 3) if r == r else None}


def main():
    df = load_raw("data/raw_tweets.csv")
    pairs = build_pairs(df)
    golden = pd.read_csv("eval/golden_set.csv")

    retriever = Retriever(pairs)
    agent = SupportAgent(retriever)
    trivial = TrivialBaseline(pairs)
    simple = SimpleBaseline(pairs)

    # --- main system ---
    def main_predict_intent(text):
        return agent.handle(text)["intent"]

    def main_predict_escalate(text, _intent_unused):
        return agent.handle(text)["escalate"]

    # NOTE: calling agent.handle() twice per example (once for intent, once for
    # escalate) is wasteful but keeps this eval harness simple and stateless;
    # see decision_log.md. For 184 examples in mock mode this is instant;
    # in live mode this doubles API spend, which is exactly the kind of thing
    # the "next week" section flags as worth fixing before production use.
    main_metrics, main_intents, main_escalates = eval_intent_and_escalation(
        "main_system", main_predict_intent, main_predict_escalate, golden)

    trivial_metrics, _, _ = eval_intent_and_escalation(
        "trivial_baseline", trivial.predict_intent, trivial.predict_escalate, golden)

    simple_metrics, _, _ = eval_intent_and_escalation(
        "simple_baseline", simple.predict_intent, simple.predict_escalate, golden)

    # --- reply quality (auto-handled only) ---
    main_replies = []
    for _, row in golden.iterrows():
        result = agent.handle(row["customer_text"])
        if not result["escalate"]:
            main_replies.append((row["customer_text"], result["draft_reply"]))
    main_quality = score_replies("main_system", main_replies)

    simple_replies = []
    for _, row in golden.iterrows():
        intent = simple.predict_intent(row["customer_text"])
        if not simple.predict_escalate(row["customer_text"], intent):
            simple_replies.append((row["customer_text"], simple.generate_reply(row["customer_text"], intent)))
    simple_quality = score_replies("simple_baseline", simple_replies)

    # trivial baseline never auto-replies, so no quality score
    trivial_quality = {"system": "trivial_baseline", "mean_judge_score": None, "n_replies_scored": 0}

    # --- judge-human agreement (main system only) ---
    agreement = judge_human_agreement(golden, agent, n=40)

    summary = {
        "classification_and_escalation": [main_metrics, simple_metrics, trivial_metrics],
        "reply_quality": [main_quality, simple_quality, trivial_quality],
        "judge_human_agreement_main_system": agreement,
        "mode": "LIVE (real Claude API)" if __import__("src.llm_client", fromlist=["USE_LIVE"]).USE_LIVE else "MOCK (no ANTHROPIC_API_KEY set — rule-based stand-in, see llm_client.py)",
    }

    with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{RESULTS_DIR}/metrics_summary.md", "w") as f:
        f.write(f"# Results ({summary['mode']})\n\n")
        f.write("## Intent classification + escalation decision (vs golden set, n={})\n\n".format(len(golden)))
        f.write("| System | Intent Acc | Intent macro-F1 | Escalation P | Escalation R | Escalation F1 | Costly FN |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m in summary["classification_and_escalation"]:
            f.write(f"| {m['system']} | {m['intent_accuracy']} | {m['intent_macro_f1']} | {m['escalation_precision']} | {m['escalation_recall']} | {m['escalation_f1']} | {m['costly_false_negatives']} |\n")
        f.write("\n## Reply quality (LLM-judge, 1-5, auto-handled replies only)\n\n")
        f.write("| System | Mean judge score | N replies scored |\n|---|---|---|\n")
        for q in summary["reply_quality"]:
            f.write(f"| {q['system']} | {q['mean_judge_score']} | {q['n_replies_scored']} |\n")
        f.write("\n## Judge-human agreement (main system, n={})\n\n".format(agreement["n"]))
        f.write(f"- Quadratic-weighted Cohen's kappa: {agreement['quadratic_weighted_kappa']}\n")
        f.write(f"- Pearson r: {agreement['pearson_r']}\n")

    print(json.dumps(summary, indent=2))
    print("\nWrote results/metrics_summary.md and results/metrics_summary.json")


if __name__ == "__main__":
    main()
