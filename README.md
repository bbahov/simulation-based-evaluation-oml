# Simulation-Based Evaluation of Online Machine Learning Models

Code companion for the paper *"A Taxonomy of Simulation-Based Evaluation for Online Machine Learning"*.

## Overview

This repository contains two evaluation scripts that demonstrate the difference between **passive** and **active** simulation environments for evaluating online recommendation models under concept drift.

Both scripts use the same model, the same oracle, the same users, and the same concept drift event. The only variable that changes is **whether the model controls which data it learns from**.

| Script | Data selection | Feedback loop |
|---|---|---|
| `active_evaluation.py` | Model picks items (epsilon-greedy) | Yes — actions shape training data |
| `passive_evaluation.py` | Items drawn at random | No — data is independent of model |

## Quick Start

### Local (terminal)

```bash
pip install -r requirements.txt
python active_evaluation.py
python passive_evaluation.py
python plot_comparison.py
```

### Google Colab

1. Upload all `.py` files to the Colab session or clone this repo
2. In the first cell: `!pip install river pandas matplotlib`
3. Run each evaluation script with `!python active_evaluation.py` etc., or paste the code directly into cells

## Output

Results are saved to `results/`:

- `active_results.csv` — per-step accuracy log from the active evaluation
- `passive_results.csv` — per-step accuracy log from the passive evaluation
- `comparison_plot.png` — side-by-side figure used in the paper

## Experimental Design

**Task:** Context-dependent item recommendation (2 users × 2 time contexts × 7 items).

**Model:** `ContextualFunkMF` — a matrix factorisation model (River's `FunkMF`) extended with a user-context composite key.

**Concept drift:** At step 2000, two of the four user-context preferences change silently. The model is not informed.

**Metric:** Rolling accuracy (window = 50) — at each step, does the model's greedy top recommendation match the oracle's current preferred item?

## Citation

If you use this code, please cite:

```
[citation placeholder]
```

## License

MIT
