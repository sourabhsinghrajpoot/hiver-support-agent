# AI Support Agent — @AmazonHelp (Hiver SDE Intern take-home)

An AI agent that, for one brand from the Customer Support on Twitter dataset,
classifies inbound customer messages, drafts a grounded reply, and decides
auto-handle vs escalate-to-human with a stated reason.

**Read `REPORT.md` first** — especially "What's misleading about my headline
number." This README is reproduction instructions only.

## ⚠️ Data note (important, read before judging any number)

This was built in a sandboxed environment with no network access to Kaggle or
HuggingFace. `data/generate_synthetic_data.py` produces a **synthetic
stand-in** with the real dataset's exact schema, noise profile, and thread
structure, modeled on `@AmazonHelp`. Full detail and how to swap in the real
CSV: `data/README_DATA.md`. Every number in `results/` is generated on this
synthetic data — treat all metrics as a **pipeline demonstration**, not a
claim about real-world accuracy. This is the single biggest caveat in the
whole project and is discussed at length in `REPORT.md`.

## Reproduce in under 15 minutes

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
bash run_all.sh
```

This runs in **MOCK mode** by default — no API key required, takes seconds,
and gives fully deterministic results (`src/llm_client.py` swaps in a
rule-based stand-in for every LLM call: classification, generation, judging).

To reproduce in **LIVE mode** (real Claude API for classification, reply
generation, and judging):
```bash
export ANTHROPIC_API_KEY=sk-...
bash run_all.sh
```
Live mode costs a small amount in API credits and takes a few minutes for
184 golden examples (3 API calls each: classify, generate, judge).

## What gets produced
- `data/raw_tweets.csv` — synthetic subsample (1500 conversation pairs)
- `eval/golden_set.csv` — 184 hand-labeled examples (see `eval/build_golden_set.py` docstring for sampling/labeling method)
- `results/metrics_summary.md` / `.json` — headline metrics: main system vs. trivial baseline vs. simple baseline
- `results/judge_human_subset.csv` — 40 examples with both judge and independent human scores, for judge-agreement evidence

## Repo layout
```
data/            synthetic data generator + provenance note
src/             the actual agent: intents.py, retrieval.py, reply_gen.py, escalation.py, pipeline.py, llm_client.py
eval/            golden set builder, baselines, LLM judge, eval harness
results/         generated metrics (committed so you can see them without running anything)
REPORT.md        problem framing, results, failure analysis, "what's misleading", next steps
decision_log.md  10-15 non-obvious decisions and why
```

## Try a single message interactively
```python
from src.data_prep import load_raw, build_pairs
from src.retrieval import Retriever
from src.pipeline import SupportAgent

pairs = build_pairs(load_raw("data/raw_tweets.csv"))
agent = SupportAgent(Retriever(pairs))
print(agent.handle("@AmazonHelp my order #445521 arrived damaged, box was crushed"))
```

## What I'd need from you to make this real
Kaggle/HuggingFace network access (or the raw CSV dropped into `data/`) and
an `ANTHROPIC_API_KEY` — the code needs zero changes for either.
