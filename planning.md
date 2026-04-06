# Planning: Reasoning Trace Length as a Proxy for Robustness

## Motivation & Novelty Assessment

### Why This Research Matters
Reasoning traces are increasingly used as a control knob for inference quality, but practitioners often optimize either for shortest possible answers (cost/latency) or for maximal verbosity (perceived reasoning quality) without robust evidence under distribution shift. Understanding the true relationship between trace length and OOD performance matters for building reliable, cost-effective LLM systems in high-stakes reasoning tasks. The outcome directly informs inference-time policies for production agents where robustness, not just in-distribution accuracy, is the primary objective.

### Gap in Existing Work
From `literature_review.md`, prior work strongly supports chain-of-thought and test-time compute, but often confounds trace length with other factors (model scale, search strategy, verifier quality, or prompt family). Recent distribution-focused work (e.g., CoT-as-mirage framing) shows OOD fragility yet does not provide a simple controlled sweep of trace length budgets with matched settings and explicit uncertainty reporting. Adaptive trace control is discussed conceptually, but undercompared against strong fixed-length baselines at similar token budgets.

### Our Novel Contribution
We run a controlled, same-model study that explicitly manipulates reasoning trace-length constraints (`none`, `short`, `medium`, `long`) and compares them to an uncertainty-triggered adaptive policy under shared evaluation conditions. We jointly evaluate: (1) answer accuracy on ID and OOD sets, (2) robustness gap, (3) efficiency (token usage), and (4) uncertainty expression/calibration behavior. This directly tests non-monotonicity and whether adaptive length outperforms fixed policies.

### Experiment Justification
- Experiment 1: Fixed trace-length sweep on ID and OOD datasets.
  - Why needed: isolates whether longer traces monotonically improve generalization.
- Experiment 2: Adaptive trace-length policy conditioned on self-reported uncertainty.
  - Why needed: tests whether dynamic allocation of reasoning budget beats fixed regimes.
- Experiment 3: Calibration/uncertainty analysis by condition.
  - Why needed: evaluates whether verbosity changes uncertainty quality, not only accuracy.
- Experiment 4: Cost-performance comparison at matched token budgets.
  - Why needed: ensures practical relevance and avoids recommending brittle expensive policies.

## Research Question
Does reasoning trace length have a non-monotonic effect on OOD robustness in LLMs, and can uncertainty-adaptive trace control improve robustness and calibration relative to fixed-length strategies?

## Background and Motivation
CoT and test-time compute methods improve many reasoning benchmarks, but evidence increasingly shows transfer failures when distribution shifts in format/composition/task type. We need a direct controlled study on trace-length budgets themselves. If the hypothesis holds, practitioners should use adaptive, uncertainty-aware reasoning depth rather than universally shorter or longer traces.

## Hypothesis Decomposition
- H1 (Non-monotonicity): OOD accuracy as a function of trace length is non-monotonic; extreme short and long regimes underperform an intermediate regime.
- H2 (Adaptive advantage): Uncertainty-adaptive trace control exceeds best fixed-length baseline on OOD robustness gap and/or token-normalized performance.
- H3 (Uncertainty behavior): Very long traces increase overconfidence or miscalibration relative to medium/adaptive traces.

Independent variables:
- Trace policy: `none`, `short`, `medium`, `long`, `adaptive`.
- Dataset split type: ID vs OOD.

Dependent variables:
- Exact-match accuracy.
- Robustness gap = ID accuracy - OOD accuracy.
- Average completion tokens / prompt+completion tokens.
- Calibration metrics from confidence estimates (ECE, Brier).

Controls:
- Same base model, temperature, parsing logic, and evaluation harness.
- Fixed random seed for sample selection.

Alternative explanations:
- Prompt wording effects instead of length effects.
- Parsing artifacts across datasets.
- Provider-side variability over time.
Mitigations: shared prompt skeleton, unified extraction rules, saved raw outputs and timestamps.

## Proposed Methodology

### Approach
Use real LLM API calls on local benchmark subsets from the pre-gathered resources. Enforce trace-length regimes via explicit prompt constraints and a bounded max token budget. Evaluate across one in-domain arithmetic set and multiple OOD sets spanning different reasoning types.

### Experimental Steps
1. Load local datasets (GSM8K, ARC-Challenge, BBH tasks, MATH-500) and create standardized QA records.
   - Rationale: heterogeneous OOD stress across math, science MCQ, and symbolic reasoning.
2. Build evaluation sample: ID=GSM8K subset; OOD={ARC, BBH-date, BBH-logic, BBH-arithmetic, MATH-500 subset}.
   - Rationale: controlled but feasible runtime with enough examples for inference.
3. Implement prompt policies:
   - `none`: direct final answer only.
   - `short`: <=2 concise reasoning steps.
   - `medium`: ~4-6 steps.
   - `long`: detailed >=8-step reasoning.
   - `adaptive`: first short attempt with confidence; if below threshold, rerun medium/long.
4. Execute all policies over same examples; log outputs, usage tokens, confidence, parsed answers.
5. Compute metrics and statistical tests; generate visualizations and tables.
6. Perform error analysis on representative failures.

### Baselines
- Direct answer (`none`) as minimal-compute baseline.
- Fixed-length `short`, `medium`, `long` as controlled interventions.
- Adaptive policy as proposed method.

### Evaluation Metrics
- Accuracy (exact match / normalized match).
- Robustness gap (ID-OOD).
- Token cost (avg prompt, completion, total).
- Cost-normalized accuracy (correct per 1k tokens).
- Uncertainty quality: ECE (10 bins), Brier score, confidence-accuracy correlation.

### Statistical Analysis Plan
- Significance level α = 0.05.
- Paired bootstrap CIs for accuracy differences (policy vs policy) on shared samples.
- McNemar test for paired correctness differences where applicable.
- Non-parametric Wilcoxon signed-rank for per-example confidence error differences.
- Multiple comparison correction via Benjamini-Hochberg FDR for pairwise policy tests.

## Expected Outcomes
- Support hypothesis if medium/adaptive outperform both short and long on OOD.
- Refute non-monotonicity if performance improves monotonically with length.
- Strong support for adaptive control if it dominates fixed policies in robustness and token efficiency.

## Timeline and Milestones
- M1 (Planning): 20 min - finalize hypotheses and design.
- M2 (Setup): 20 min - env, dependencies, metadata capture.
- M3 (Implementation): 75 min - experiment/eval scripts.
- M4 (Experiments): 75 min - run all policy conditions.
- M5 (Analysis): 40 min - stats, plots, error analysis.
- M6 (Documentation/validation): 30 min - REPORT, README, reproducibility checks.

## Potential Challenges
- API rate limits or intermittent failures.
  - Mitigation: retry with backoff, caching JSONL outputs.
- Parsing diverse answer formats.
  - Mitigation: dataset-specific normalization and extraction rules.
- Cost/time constraints for full benchmarks.
  - Mitigation: preregistered subsampling with fixed seed and transparent reporting.
- Confidence self-report bias.
  - Mitigation: treat as behavioral signal, evaluate calibration empirically.

## Success Criteria
- Full experiment run completed with real model outputs saved.
- All five policies evaluated on both ID and OOD slices.
- Statistical tests and uncertainty analyses reported with CIs/p-values.
- Reproducible pipeline and documented limitations in final report.
