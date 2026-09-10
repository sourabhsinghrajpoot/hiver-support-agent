#!/usr/bin/env bash
# Reproduces every headline number in REPORT.md in under 15 minutes.
# Runs in MOCK mode by default (no API key needed). Set ANTHROPIC_API_KEY
# in your environment first to reproduce in LIVE mode instead.
set -e
export PYTHONPATH=.

echo "== 1/3: generating synthetic subsample (data/README_DATA.md explains why) =="
python3 data/generate_synthetic_data.py --n 1500 --seed 42 --out data/raw_tweets.csv

echo "== 2/3: building golden evaluation set (150-250 hand-labeled examples) =="
python3 eval/build_golden_set.py

echo "== 3/3: running evaluation harness (main system vs 2 baselines + judge agreement) =="
python3 eval/run_eval.py

echo
echo "Done. See results/metrics_summary.md for headline numbers."
