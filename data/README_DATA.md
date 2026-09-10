# Data note (read this first)

This sandbox has no network access to Kaggle or HuggingFace (outbound access is
allowlisted to package registries only — pypi, npm, github — not kaggle.com or
huggingface.co). So the code in this repo was **not** run against the real
`thoughtvector/customer-support-on-twitter` file.

Instead `generate_synthetic_data.py` produces a **synthetic-but-structurally-identical**
subsample: same schema as the real `twcs.csv`
(`tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`),
same noise profile (typos, hashtags, @handles, emoji, truncated thoughts, multi-turn
threads), modeled on **@AmazonHelp**, one of the largest brands in the real dataset.
Ground-truth intent labels are attached at generation time, which is only possible
because the data is synthetic — this is called out explicitly in `REPORT.md` under
"what's misleading about my headline number," since it is the single biggest caveat
on every result in this repo.

## To run this against the real data (what I'd do with real access)

1. Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`).
2. Filter to `@AmazonHelp` threads: inbound tweets whose `response_tweet_id`
   resolves to a tweet authored by `AmazonHelp`.
3. Drop `generate_synthetic_data.py` from the pipeline; point
   `src/data_prep.py::load_raw()` at the real CSV — the schema already matches,
   so no other code changes are needed.
4. Re-run `eval/build_golden_set.py` — it will need **real** hand-labeling at
   that point (right now it's labeling synthetic data whose labels are already
   known, which is a shortcut documented in the report).
5. Everything downstream (`src/`, `eval/`) is data-source-agnostic.
