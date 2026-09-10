# Results (MOCK (no ANTHROPIC_API_KEY set — rule-based stand-in, see llm_client.py))

## Intent classification + escalation decision (vs golden set, n=184)

| System | Intent Acc | Intent macro-F1 | Escalation P | Escalation R | Escalation F1 | Costly FN |
|---|---|---|---|---|---|---|
| main_system | 0.957 | 0.955 | 0.633 | 0.644 | 0.639 | 21 |
| simple_baseline | 1.0 | 1.0 | 0.739 | 0.576 | 0.648 | 25 |
| trivial_baseline | 0.125 | 0.028 | 0.321 | 1.0 | 0.486 | 0 |

## Reply quality (LLM-judge, 1-5, auto-handled replies only)

| System | Mean judge score | N replies scored |
|---|---|---|
| main_system | 3.9 | 124 |
| simple_baseline | 3.5 | 138 |
| trivial_baseline | None | 0 |

## Judge-human agreement (main system, n=40)

- Quadratic-weighted Cohen's kappa: 0.402
- Pearson r: 0.477
