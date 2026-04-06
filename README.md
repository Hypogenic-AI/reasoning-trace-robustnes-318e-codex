# Reasoning Trace Length vs OOD Robustness

This project tests whether longer reasoning traces always improve generalization in LLMs. We run real `gpt-4.1` API experiments over ID/OOD reasoning datasets while systematically controlling trace length and comparing fixed policies to an uncertainty-triggered adaptive policy.

## Key Findings
- OOD accuracy was best with `adaptive` (0.800), then `medium` (0.767), while `none` was worst (0.367).
- Relationship is not "longer is always better": `long` underperformed `medium` on OOD while using far more tokens.
- Robustness gap (`ID - OOD`) improved from 0.633 (`none`) to 0.200 (`adaptive`).
- Adaptive significantly beat `none` on OOD (McNemar p=0.0019, BH q=0.0078).
- CoT-heavy policies were overconfident; calibration remained a core weakness.

## Reproduce
1. Environment:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Run experiment:
```bash
python src/experiment_trace_length.py \
  --model gpt-4.1 \
  --id-n 18 \
  --ood-n 6 \
  --temperature 0.2 \
  --adapt-threshold 0.98 \
  --out-jsonl results/raw_outputs.jsonl \
  --summary-json results/run_summary.json
```

3. Analyze and plot:
```bash
python src/analyze_trace_length.py \
  --raw-jsonl results/raw_outputs.jsonl \
  --metrics-json results/metrics.json \
  --pairwise-json results/pairwise_tests.json \
  --plots-dir results/plots
```

## File Structure
- `planning.md`: Motivation, novelty, and preregistered analysis plan.
- `src/experiment_trace_length.py`: API experiment harness and evaluation logic.
- `src/analyze_trace_length.py`: Metrics, statistical tests, visualizations.
- `results/raw_outputs.jsonl`: Per-example model outputs and metadata.
- `results/metrics.json`: Aggregated metrics and robustness summary.
- `results/pairwise_tests.json`: Pairwise significance tests.
- `results/plots/`: Generated figures.
- `REPORT.md`: Full research report with methodology, results, and limitations.

For complete details, see `REPORT.md`.
