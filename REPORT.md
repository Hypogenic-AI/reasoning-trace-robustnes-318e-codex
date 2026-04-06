# Reasoning Trace Length as a Proxy for Robustness: A Systematic Investigation

## 1. Executive Summary
This study tests whether longer reasoning traces always improve generalization, or whether there is a non-monotonic relationship between trace length and OOD robustness.

Using real `gpt-4.1` API calls over a controlled ID/OOD benchmark mix (GSM8K ID; ARC/BBH/MATH-500 OOD), fixed trace policies (`none`, `short`, `medium`, `long`) and an uncertainty-triggered adaptive policy were compared under a single evaluation harness.

Key result: on OOD data, `adaptive` achieved the best accuracy (0.800) and smallest robustness gap (0.200), while `none` collapsed to 0.367 OOD accuracy. Gains over `none` were statistically significant; gains over stronger fixed CoT baselines were positive but not significant at this sample size.

Practical implication: very short/no-trace behavior is brittle under shift; very long traces are expensive and not best-performing; adaptive trace budgeting is the strongest operating point in this experiment.

## 2. Goal

### Hypothesis
There exists a non-monotonic relationship between reasoning trace length and OOD robustness in LLMs, and uncertainty-informed adaptive trace-length control outperforms fixed-length policies for robustness-efficiency trade-offs.

### Why This Is Important
Inference-time trace policy is a deployment knob that directly affects reliability, cost, and latency. Teams often optimize only for short outputs or verbose CoT without robust OOD evidence.

### Problem Solved
This work provides a controlled, same-model evaluation of trace-length constraints and adaptive escalation, separating policy effects from model or dataset changes.

### Expected Impact
Results inform production policy selection for reasoning systems where OOD behavior and calibration matter.

## 3. Data Construction

### Dataset Description
Sources (pre-downloaded in `datasets/`):
- GSM8K (`datasets/gsm8k_main`) for ID arithmetic reasoning.
- ARC-Challenge (`datasets/ai2_arc_challenge`) for science MCQ OOD.
- BBH tasks (`datasets/bbh_date_understanding`, `datasets/bbh_logical_deduction_three_objects`, `datasets/bbh_multistep_arithmetic_two`) for OOD reasoning variants.
- MATH-500 (`datasets/math500`) for harder math OOD.

Evaluation subset (seed=42):
- ID: 18 GSM8K test examples.
- OOD: 6 examples from each OOD dataset (30 total).
- Total examples: 48.

### Example Samples
| Dataset | Question (truncated) | Gold |
|---|---|---|
| GSM8K | "Jared is trying to increase his typing speed..." | `52` |
| ARC | "An astronomer observes that a planet rotates faster..." | `C` |
| BBH date | "Today is Christmas Eve of 1937..." | `(B)` |

### Data Quality
From `results/data_summary.json`:
- Missing question values: 0.0%
- Missing gold values: 0.0%
- Duplicate IDs in evaluation slice: 0
- Answer types: numeric 24, choice 18, text 6

### Preprocessing Steps
1. Loaded Arrow datasets with `datasets.load_from_disk`.
2. Normalized gold answers by dataset:
   - GSM8K: parse `####` final value.
   - Choice tasks: parse option letter `(A-F)`.
   - Numeric tasks: first numeric extraction.
   - MATH-500: normalized string match fallback.
3. Constructed a unified schema: `(dataset, split_type, qid, question, gold, answer_type)`.

### Train/Val/Test Splits
No model training was performed; this is pure inference-time evaluation.
- ID slice sampled from benchmark test set.
- OOD slice sampled from other benchmark test sets.

## 4. Experiment Description

### Methodology
#### High-Level Approach
Run identical examples through 5 trace policies using one model (`gpt-4.1`) with fixed temperature and parser:
- `none`: final answer only.
- `short`: <=2 concise steps.
- `medium`: ~4-6 steps.
- `long`: >=8 detailed steps.
- `adaptive`: short probe + confidence; escalate to long when confidence < threshold.

#### Why This Method
This directly manipulates trace length while holding model and dataset constant. It operationalizes adaptive trace control without requiring fine-tuning.

Alternatives considered:
- Multi-model comparison (rejected for confound reduction).
- Full-benchmark exhaustive runs (deferred for cost/runtime).

### Implementation Details
#### Tools and Libraries
From `results/environment.json`:
- Python 3.12.8
- openai 2.30.0
- datasets 4.8.4
- numpy 2.4.4
- pandas 3.0.2
- scipy 1.17.1
- statsmodels 0.14.6
- matplotlib 3.10.8
- seaborn 0.13.2

#### Algorithms/Models
- Model: `gpt-4.1` (real API calls).
- Output contract: strict JSON (`reasoning`, `final_answer`, `confidence`, `uncertainty_note`).
- Evaluation: exact/normalized matching by answer type.

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| model | `gpt-4.1` | current recommended frontier API |
| temperature | 0.2 | low-variance inference |
| seed | 42 | reproducibility |
| ID sample size | 18 | runtime-constrained preregistered subset |
| OOD sample size per dataset | 6 | balanced OOD coverage |
| adaptive threshold | 0.98 | tuned to ensure real escalation behavior |
| max output tokens (`none`) | 80 | policy budget |
| max output tokens (`short`) | 180 | policy budget |
| max output tokens (`medium`) | 320 | policy budget |
| max output tokens (`long`) | 700 | policy budget |

### Experimental Protocol
#### Reproducibility Information
- Number of runs per policy: 1 pass over all examples.
- Random seed: 42.
- Hardware:
  - GPUs detected: 2x NVIDIA GeForce RTX 3090 24GB (`results/gpu_info.csv`).
  - Note: API-based inference; GPU not used for model inference.
- End-to-end runtime (48 examples x 5 policies): ~6 minutes per full run.

#### Evaluation Metrics
- Accuracy: fraction correct.
- Robustness gap: `ID accuracy - OOD accuracy`.
- Token cost: average total tokens.
- Calibration: Brier score and ECE (10 bins) from self-reported confidence.
- Pairwise hypothesis testing: paired bootstrap CI and McNemar p-values (BH corrected).

### Raw Results
#### Main Table (ID/OOD)
| Policy | ID Acc | OOD Acc | Robustness Gap | Avg Tokens (OOD) | OOD ECE |
|---|---:|---:|---:|---:|---:|
| none | 1.000 | 0.367 | 0.633 | 212.0 | 0.2167 |
| short | 1.000 | 0.700 | 0.300 | 234.1 | 0.2663 |
| medium | 1.000 | 0.767 | 0.233 | 299.6 | 0.1947 |
| long | 1.000 | 0.700 | 0.300 | 381.4 | 0.2840 |
| adaptive | 1.000 | **0.800** | **0.200** | 236.7 | **0.1823** |

#### Pairwise OOD Tests (adaptive vs fixed)
| Comparison | Accuracy Diff | 95% Bootstrap CI | McNemar p | BH q |
|---|---:|---|---:|---:|
| adaptive vs none | +0.433 | [0.233, 0.633] | 0.0019 | 0.0078 |
| adaptive vs short | +0.100 | [-0.033, 0.267] | 0.3711 | 0.4948 |
| adaptive vs medium | +0.033 | [-0.100, 0.167] | 1.0000 | 1.0000 |
| adaptive vs long | +0.100 | [-0.033, 0.267] | 0.3711 | 0.4948 |

#### Dataset-level OOD Accuracy Snapshot
- ARC-Challenge: `none=0.000`, others mostly high.
- BBH date_understanding: generally hardest across policies.
- MATH-500 subset: weak for all policies; `none=0.000`, others ~0.5.

#### Output Locations
- Raw outputs: `results/raw_outputs.jsonl`
- Run metadata: `results/run_summary.json`
- Metrics: `results/metrics.json`
- Pairwise tests: `results/pairwise_tests.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. `none` degrades sharply OOD (0.367), confirming that minimal/no-trace prompting is brittle under shift.
2. `medium` and `adaptive` outperform `short` and `long` on OOD, consistent with a non-extreme optimum.
3. `long` has the highest token cost and does not beat `medium` or `adaptive`.
4. `adaptive` is best overall on OOD with lowest robustness gap and best OOD ECE.

### Hypothesis Testing Results
- H1 (non-monotonic tendency): supported directionally.
  - Fixed-policy OOD scores: `none 0.367 -> short 0.700 -> medium 0.767 -> long 0.700` (peak then drop).
  - Kendall tau over fixed length ranks: `tau=0.548`, `p=0.279` (insufficient power for strict significance).
- H2 (adaptive advantage): partially supported.
  - Adaptive is top OOD performer; significantly better than `none`.
  - Versus `medium`/`short`/`long`, improvements are positive but not significant at current N.
- H3 (uncertainty behavior): supported.
  - Long traces show poor OOD calibration despite high confidence (`ECE 0.284`, mean conf ~0.984).

### Visualizations
Generated in `results/plots/`:
- `ood_accuracy_by_policy.png`
- `robustness_gap_by_policy.png`
- `token_usage_by_policy_split.png`
- `confidence_vs_accuracy.png`

### Surprises and Insights
- All policies scored 1.0 on this ID slice (ceiling effect), making robustness gap mainly driven by OOD behavior.
- Self-reported confidence is highly inflated for CoT-heavy modes; calibration remains weak even when accuracy rises.
- Adaptive second-pass rate was low (~10.4%), indicating confidence thresholding can save tokens while retaining strong OOD results.

### Error Analysis
Representative failures (adaptive OOD):
- BBH date questions with precise temporal offsets (often overconfident wrong date).
- BBH logical deduction where answer format drifted from option letter to descriptive phrase.
- MATH-500 symbolic expressions where exact-form normalization is brittle.

### Limitations
- Small sample size (48 questions) limits statistical power.
- Single model/provider and single temperature setting.
- Confidence is self-reported by the model, not externally calibrated.
- MATH-500 text-match evaluation is strict and may undercount semantically equivalent answers.

## 6. Conclusions
Longer traces are not uniformly better. In this controlled run, OOD performance improved from no-trace to medium reasoning, then declined at very long traces, while adaptive policy achieved the best robustness-cost balance.

Practically, teams should avoid defaulting to either minimal or maximal verbosity. A confidence-triggered adaptive strategy is a stronger default, with explicit calibration monitoring.

Confidence level in findings: moderate. The directional pattern is clear, but larger-scale repeated runs are needed for stronger significance claims across all pairwise comparisons.

## 7. Next Steps

### Immediate Follow-ups
1. Increase sample sizes (e.g., >=200 OOD examples) for stronger hypothesis tests.
2. Add repeated runs/seeds and temperature sweeps to quantify variance.
3. Evaluate alternative adaptive triggers (entropy/logprob-based rather than self-reported confidence).

### Alternative Approaches
- Verifier-guided selection (PRM/ORM reranking) to disentangle trace quality from length.
- Budget-matched search methods (e.g., slimmed self-consistency) for stronger adaptive baselines.

### Broader Extensions
- Extend to additional OOD axes: formatting perturbations, compositional depth shifts, adversarial paraphrases.
- Compare model families (GPT-5 class, Claude Sonnet 4.5, Gemini 2.5 Pro) under identical protocol.

### Open Questions
- Which uncertainty signals are most reliable for escalation triggers?
- Does optimal trace length depend more on task type or on latent instance difficulty?
- How do these effects evolve at larger context windows and higher-capability models?

## References
- See `literature_review.md` and `papers/` for the reviewed bibliography used in planning.
