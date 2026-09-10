# Report — AI Support Agent for @AmazonHelp

## 0. Read this first
Every number below comes from a **synthetic, schema-identical stand-in** for
the real Kaggle dataset (no Kaggle/HuggingFace network access in this
sandbox — see `data/README_DATA.md`) run in **MOCK LLM mode** (no API key in
this environment — see `src/llm_client.py`). Treat this report as evidence
that the *pipeline and evaluation methodology* work end to end, not as a
claim about real-world accuracy. Section 4 goes into exactly why.

---

## 1. Problem framing

**What "good" means for this brand.** @AmazonHelp's job on Twitter is damage
control on a small number of recurring, high-volume issues (where's my
order, my package is damaged, refund my return) plus a small number of
high-stakes issues (account takeover, billing fraud) where the cost of a
wrong automated answer is much higher than the cost of a slow human answer.
"Good" therefore means: **near-perfect precision on the decision to
auto-handle** (a wrong auto-reply on a real complaint damages trust and
sometimes money) even at the cost of recall — over-escalating a few
easy cases is cheap; under-escalating a hard one is not.

**What I chose not to build:**
- **Multi-turn conversation state.** Real threads continue past the first
  reply (customer pushes back, brand replies again). I collapse each thread
  to (first customer message → first brand reply). A production system needs
  full thread state; this is flagged, not hidden (decision log #13).
- **Fine-grained NLU-style intents (Banking77-level granularity).** I used 8
  intents mapped to distinct resolution playbooks, not 77 fine-grained
  categories — see decision log #3.
- **A learned/trained escalation model.** Escalation is a hand-written,
  auditable rule layer (decision log #4), not something I let an LLM decide
  end-to-end.
- **An embeddings-based retriever.** TF-IDF only (decision log #5).
- **Real hand-labeling by a second human.** Judge-human agreement uses an
  independently-coded heuristic standing in for a second rater — flagged
  explicitly in `eval/human_label_subset.py` and again in section 4 below.

---

## 2. Results vs. baselines

Three systems evaluated on the 184-example golden set:
- **Trivial baseline**: always predicts the majority intent, always escalates, never replies.
- **Simple baseline**: TF-IDF + Logistic Regression intent classifier (no LLM), fixed per-intent template reply, static escalation rule (escalate iff intent ∈ {account_access, billing_dispute}).
- **Main system**: LLM intent classification + TF-IDF retrieval-grounded LLM reply generation + rule-based escalation with confidence/similarity floors.

| System | Intent Acc | Intent macro-F1 | Escalation P | Escalation R | Escalation F1 | Costly FN* |
|---|---|---|---|---|---|---|
| main_system | 0.957 | 0.955 | 0.633 | 0.644 | 0.639 | 21 |
| simple_baseline | 1.000 | 1.000 | 0.739 | 0.576 | 0.648 | 25 |
| trivial_baseline | 0.125 | 0.028 | 0.321 | 1.000 | 0.486 | 0 |

*Costly FN = auto-handled a message the golden set says should have escalated — the expensive error class.

| System | Mean judge score (1-5) | N replies scored |
|---|---|---|
| main_system | 3.90 | 124 |
| simple_baseline | 3.50 | 138 |
| trivial_baseline | — (never auto-replies) | 0 |

**Judge-human agreement (main system, n=40):** quadratic-weighted Cohen's κ = 0.402, Pearson r = 0.477 — moderate agreement. Not strong enough to trust the judge alone in production; treat judge scores as a triage signal, spot-check the rest by hand. Full scored pairs in `results/judge_human_subset.csv`.

**Reading these numbers correctly:** the main system beats the trivial
baseline everywhere and produces higher-quality replies than the simple
baseline (3.90 vs 3.50), but **loses to the simple baseline on intent
accuracy (0.957 vs 1.000) and has more costly false negatives (21 vs 25 is
actually better, not worse — main system is fewer)**. The simple baseline's
perfect intent accuracy is itself a red flag, not a win — see section 4.

---

## 3. Failure analysis — top 5 failure modes (real examples from the run)

**1. Ambiguous "delivery vs. status" boundary in the mock classifier.**
`"never received order 316356, tracking says delivered but nothing here"` was
classified `order_status` (conf 0.70) when the gold label is `delivery_issue`.
Hypothesis: the mock classifier's keyword list has "tracking" mapped to
`order_status`, and this message uses "tracking" in a delivery-failure
context. A real LLM would likely catch "delivered but nothing here" as the
stronger signal; keyword-based classification can't weigh context. This
exact failure recurred 8/8 times for this template — it's systematic, not
noise.

**2. Damaged-package messages auto-handled when they should escalate.**
`"my package for order 898867 arrived DAMAGED, box was crushed #fail"` →
intent `delivery_issue` (conf 0.85), similarity ~1.0, auto-handled with
*"I've escalated order 898867 to our carrier investigation team..."* This is
actually a reasonable reply, but the golden label says this case should go
to a human (repeat-damage pattern, or emotionally charged customer). Hypothesis:
my confidence/similarity floors (0.55 / 0.15) are tuned for "is this reply
safe to send," not "is this customer upset enough to need a human touch" —
those are different signals and my rule layer only checks the first one.
21 such costly false negatives total; see decision log #15 for why the eval
was designed to surface exactly this tension.

**3. Perfect-looking baseline accuracy that's actually overfitting to
template artifacts.** The simple baseline hits 100% intent accuracy — on
synthetic data generated from a small, fixed set of sentence templates, a
TF-IDF+LogReg classifier can essentially memorize surface n-grams. This
isn't a real 100% — it's the single clearest artifact of using synthetic
data, and it's the headline "misleading number" (section 4).

**4. Sensitive-keyword rule is brittle to phrasing.** `"unauthorized charge"`
and `"this is fraud"` correctly trigger escalation, but a customer who says
*"charged me for something I returned, that's basically stealing"* would
slip past the keyword list entirely. Hypothesis: a static keyword list is a
reasonable v1 guardrail but will always have false negatives against
adversarial or just differently-worded phrasing; needs either a broader list
or a second LLM-based safety classifier as a backstop (see section 5).

**5. Judge-human agreement is only moderate (κ=0.40).** Both the mock judge
and the mock "human" are heuristics I wrote, not two independent real
people, so this number is a methodology demonstration, not real evidence of
judge reliability. Hypothesis for why it's not higher even so: both graders
reward "mentions an order id" and "apologetic tone" but disagree on how much
to penalize short-but-correct replies, which is exactly the kind of
disagreement two real human raters would also have — a sign the eval design
is at least stress-testing something real, but 40 examples is too few to be
confident about the exact number.

---

## 4. What's misleading about my headline number

Pick almost any number in section 2 and it overstates real-world readiness:

- **All numbers are on synthetic data.** The 8 intents, their relative
  frequencies, and even the sentence templates were written by me. A
  classifier or judge that does well here has learned my synthetic
  distribution, not real @AmazonHelp customer language, which is messier,
  more multilingual, and less template-shaped than anything here.
- **The simple baseline's 100% intent accuracy is the biggest tell.**
  On real, noisy data I would expect the LLM-based main system to *beat* the
  TF-IDF baseline, not lose to it — the reversal here is diagnostic of
  synthetic-data overfitting, not of the simple baseline being genuinely
  better.
- **MOCK-mode LLM calls are keyword heuristics, not a real language model.**
  Every "LLM" number in section 2 — classification, reply generation,
  judging — was produced by `src/llm_client.py`'s rule-based fallback
  (no API key in this sandbox). The architecture is real and swaps to a real
  Claude call with one environment variable, but **no number in this report
  should be read as "how good is Claude at this task."**
- **The golden set's escalation labels are one person's judgment call**
  (mine), applied a few minutes after writing the escalation rules — there's
  an obvious risk I unconsciously labeled examples to match rules I'd
  already written, which would make the escalation-quality numbers look
  better than they'd be against an independently-labeled set.
- **Judge-human agreement (κ=0.40) is itself between two heuristics I wrote**,
  not two real people — see failure mode 5. Read it as "the eval harness
  computes agreement correctly," not "the judge is validated."
- **184 golden examples is small.** Escalation precision/recall in
  particular is computed over ~30-45 positive examples per system — a
  handful of different labeling calls would move these numbers several
  points.
- **N replies scored differs across systems** (124 vs 138) because systems
  disagree on what to escalate — the "mean judge score" comparison is on
  overlapping-but-not-identical subsets, not a clean apples-to-apples set.

---

## 5. What I'd do next with one more week

1. **Get real data.** Download the actual Kaggle CSV and re-run everything —
   the pipeline needs zero code changes (`data/README_DATA.md`); only
   `eval/build_golden_set.py`'s labeling step needs real human labeling.
2. **Get a real API key and re-run in LIVE mode**, then re-do sections 2-4
   with real Claude classification/generation/judging, since that's the
   actual product being pitched.
3. **Get a second real human labeler** for both the golden set and the
   judge-agreement subset, and report real inter-rater agreement before
   trusting the golden labels themselves.
4. **Add an embeddings retriever alongside TF-IDF** and A/B the two on
   reply-quality judge scores, now that there'd be enough real vocabulary
   diversity for it to matter (decision log #5).
5. **Add a second, LLM-based safety classifier as a backstop** to the static
   sensitive-keyword list (failure mode 4), specifically targeting
   fraud/security language that doesn't hit the keyword list.
6. **Multi-turn thread modeling** — a large fraction of real threads involve
   a customer follow-up after the first brand reply; v1 ignores this
   entirely (decision log #13).
7. **Cache/dedupe LLM calls in the eval harness** — `run_eval.py` currently
   calls `agent.handle()` twice per golden example (decision log #14); fine
   for 184 mock-mode examples, wasteful and slow against a real API and a
   real-sized eval set.
