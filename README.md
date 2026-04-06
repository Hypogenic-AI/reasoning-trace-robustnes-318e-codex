# Reasoning Trace Length as a Proxy for Robustness

Does a longer reasoning chain always mean better generalization? We systematically vary reasoning trace lengths in LLMs and measure how this influences out-of-distribution (OOD) performance and uncertainty expression.

## Key Findings

- **Non-monotonicity confirmed:** Short/medium reasoning (1-6 steps) achieves 90% ID accuracy; long reasoning (8+ steps) drops to 80%. Effect is stronger on OOD data.
- **Verbose reasoning harms OOD robustness:** On MATH-500 (OOD-near), short traces achieve 46-70% accuracy vs. 15-40% for long traces across two model sizes.
- **Adaptive trace control is promising:** Uncertainty-adaptive policy achieves best OOD-near accuracy (56%) with smallest robustness gap (30%).
- **Concise reasoning dominates on efficiency:** Short reasoning is 70x more token-efficient than long reasoning with equal or better accuracy.
- **Confidence calibration is poor:** Self-reported confidence (mean 0.924) far exceeds actual accuracy (0.644), limiting adaptive policy effectiveness.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install openai datasets numpy scipy matplotlib seaborn tqdm
export OPENAI_API_KEY=<your-key>

python src/load_datasets.py        # Load 250 questions from 5 benchmarks
python src/run_experiment.py       # Run 5 policies x 250 items (GPT-4.1-nano)
python src/run_confirmatory.py     # Confirmatory run (GPT-4.1-mini)
python src/analyze_results.py      # Generate tables, plots, statistics
```

## Project Structure

```
src/
  load_datasets.py         # Dataset loading (GSM8K, MATH-500, ARC, MMLU, CSQA)
  run_experiment.py        # Main experiment: 5 trace-length policies
  run_confirmatory.py      # Confirmatory experiment with stronger model
  analyze_results.py       # Statistical analysis and visualization

results/
  raw/                     # Raw experiment outputs (JSONL)
  plots/                   # Generated visualizations
  summary.json             # Aggregated metrics

datasets/                  # Pre-downloaded benchmarks (HuggingFace format)
papers/                    # Related research papers (PDF)
```

## Experiment Design

5 trace-length policies tested on 250 questions across 5 benchmarks:

| Policy | Description | Avg Tokens |
|--------|------------|-----------|
| none | Direct answer, no reasoning | 9 |
| short | 1-2 brief reasoning steps | 85 |
| medium | 4-6 step chain-of-thought | 315 |
| long | 8+ detailed steps with verification | 766 |
| adaptive | Short first; retry long if confidence < 70% | 185 |

See [REPORT.md](REPORT.md) for full results, statistical tests, and analysis.
