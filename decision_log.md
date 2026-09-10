# Decision log

Non-obvious decisions made while building this, and why.

1. **Brand: @AmazonHelp, not @AppleSupport.** Amazon's issues (order status,
   delivery, refunds) have a clear "correct resolution" that generalizes
   across customers, so grounding-by-retrieval works well. Apple's issues
   often need device-specific diagnostics that a Twitter reply can't actually
   resolve, which would make "reply quality" mostly about tone, not substance.

2. **Synthetic data instead of skipping the assignment.** No Kaggle/HF network
   access in this sandbox. Rather than either faking real-looking numbers or
   refusing to produce a working system, I built a schema-identical synthetic
   generator and flagged this everywhere (data/README_DATA.md, README.md,
   REPORT.md). A real submission would swap in the actual CSV with zero code
   changes downstream of `data_prep.py`.

3. **8 intents, not Banking77's 77.** Twitter support replies map to a small
   set of *resolution playbooks* (refund, replacement, DM-for-security,
   etc.) — that's the unit that matters for auto-handle vs escalate, not
   fine-grained NLU categories. More intents would fragment training/retrieval
   data without changing what action gets taken.

4. **Rule-based escalation layer sits ON TOP of the LLM, not inside it.**
   I didn't ask the LLM "should this escalate?" as a free-form judgment.
   Escalation is safety-critical and needs to be auditable by a human who
   isn't debugging a prompt — `src/escalation.py` is ~30 lines anyone on the
   team can read and edit without touching the LLM pipeline at all.

5. **TF-IDF retrieval, not embeddings.** For short, templated support tweets,
   lexical overlap ("damaged", "refund", "locked out") is almost all the
   signal, and TF-IDF is free, deterministic, and lets you literally see why
   an example was retrieved (useful for debugging bad grounding). Listed
   under "what I'd do next" as the first upgrade once volume/vocabulary grows.

6. **Single overall judge score (1-5), not multi-dimensional rubric.** With
   184 golden examples and only 40 in the judge-human agreement check,
   4 correlated sub-scores (grounding/tone/correctness/actionability) would
   add apparent precision without adding real signal, and would be harder to
   sanity-check by hand.

7. **MOCK-mode LLM by design, not as a fallback hack.** `src/llm_client.py`
   is a real architectural choice: every LLM call point (classify, generate,
   judge) is behind one function so the *whole system* is testable and
   reproducible in <15 min without an API key, and swapping to LIVE mode
   requires zero code changes elsewhere. The cost: MOCK-mode numbers are not
   a claim about a real LLM's quality (see REPORT.md).

8. **Stratified, not random, golden-set sampling.** Real brand traffic is
   dominated by order_status/delivery_issue. Random sampling would starve the
   golden set of account_access/billing_dispute examples — exactly where
   escalation mistakes are costliest. I fixed ~23 examples per intent instead.

9. **"Hand-labeling" on synthetic data means independent re-labeling, not
   trusting the generator's ground truth.** I manually reviewed every golden
   example and assigned my own escalate/no-escalate judgment using a written
   rule (money/security/safety/ambiguity → escalate), which disagreed with
   the generator's synthetic label ~6-9% of the time — those disagreements
   are kept, not smoothed over, because real annotators disagree with
   "ground truth" too.

10. **Judge-human agreement uses an independently-coded heuristic, not a
    second reading of the judge's own rationale.** `human_label_subset.py`
    deliberately looks at different signals (urgency-language mismatch,
    action-verb presence) than the mock judge does, plus injected noise, so
    the reported kappa/Pearson-r isn't trivially inflated by both scorers
    using the same cues. Still a stand-in for a real second human — flagged
    as a limitation, not presented as a real inter-rater study.

11. **Costly false negatives reported as a standalone number, not buried in
    F1.** An auto-handled message that should have been escalated is the
    single most expensive failure mode for this product. Escalation F1 can
    look fine while this number is high; I report it separately so it can't
    hide.

12. **Reply quality only scored for auto-handled messages.** Baselines/system
    calls that escalate produce no reply, so scoring "no reply" as quality-0
    would conflate "chose not to answer" with "answered badly" — two very
    different failure types with different fixes.

13. **First-customer-message-only threading (no multi-turn).** The real
    dataset has threads where the customer replies again after the brand's
    first response. Reconstructing full multi-turn context is a real project
    on its own; v1 treats each thread as single-turn and says so explicitly
    rather than silently dropping the harder threads without comment.

14. **`agent.handle()` called twice per eval example (once for intent, once
    for escalation) instead of caching.** Kept the eval harness stateless and
    simple for a take-home; flagged in code comments as something to fix
    before this touches a real API bill in production.

15. **Synthetic escalate-rates were baked in per-intent at generation time**
    (e.g. billing_dispute escalates 70% of the time, order_status only 10%)
    to make the escalation-decision problem non-trivial — a dataset where
    escalation is 1:1 with intent would make the escalation classifier
    trivially perfect and wouldn't stress-test the confidence/similarity
    thresholds in `src/escalation.py` at all.
