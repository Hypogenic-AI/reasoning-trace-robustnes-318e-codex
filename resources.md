# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "Reasoning Trace Length as a Proxy for Robustness: A Systematic Investigation." Resources include 22 papers, 6 downloaded datasets (1 gated), and 6 code repositories.

## Papers

Total papers downloaded: 22

| # | Title | Authors | Year | Citations | File | Key Info |
|---|-------|---------|------|-----------|------|----------|
| 1 | Impact of Reasoning Step Length | Jin et al. | 2024 | 171 | `jin2024_impact_reasoning_step_length.pdf` | Foundational: longer steps help |
| 2 | When More is Less: CoT Length | Wu et al. | 2025 | 149 | `wu2025_when_more_is_less_cot_length.pdf` | Inverted U-shape, scaling laws |
| 3 | Between Underthinking and Overthinking | Su et al. | 2025 | 76 | `su2025_underthinking_overthinking.pdf` | Over/underthinking empirical study |
| 4 | Demystifying Long CoT | Chen et al. | 2025 | - | `chen2025_demystifying_long_cot.pdf` | Long CoT enables self-correction |
| 5 | L1: Length Control via RL | Aggarwal & Welleck | 2025 | 286 | `aggarwal2025_l1_length_control_rl.pdf` | LCPO, adaptive > fixed, OOD gen. |
| 6 | Token-Budget-Aware Reasoning | Han et al. | 2024 | - | `han2024_token_budget_aware_reasoning.pdf` | Token budget prompting |
| 7 | SelfBudgeter | Wang et al. | 2025 | - | `wang2025_selfbudgeter.pdf` | Autonomous budget prediction |
| 8 | Reasoning on a Budget (Survey) | Zhang et al. | 2025 | - | `zhang2025_reasoning_budget_survey.pdf` | L1/L2 controllability taxonomy |
| 9 | C3oT: Shorter CoT | Kang et al. | 2024 | 151 | `kang2024_c3ot_shorter_cot.pdf` | 50%+ compression w/o accuracy loss |
| 10 | TokenSkip | Xia et al. | 2025 | 192 | `xia2025_tokenskip.pdf` | Token-level CoT compression |
| 11 | Concise Reasoning / LiteCoT | Wu et al. | 2025 | 13 | `wu2025_concise_reasoning_big_gains.pdf` | Difficulty-aware CoT pruning |
| 12 | Robust Reasoning w/ Noisy Rationales | Wang et al. | 2024 | - | `wang2024_robust_reasoning_noisy_rationales.pdf` | NoRa benchmark, CD-CoT |
| 13 | CoT Reasoning as Mirage | Li et al. | 2025 | - | `li2025_cot_mirage_distribution.pdf` | CoT brittleness OOD |
| 14 | Illusion of Thinking | Shojaee et al. | 2025 | 318 | `shojaee2025_illusion_of_thinking.pdf` | 3 complexity regimes |
| 15 | Probing Reasoning Trajectories | Ballon et al. | 2026 | 0 | `ballon2026_probing_trajectories_reasoning.pdf` | Truncation probing protocol |
| 16 | Shape of Reasoning (Topological) | Tao et al. | 2025 | - | `tao2025_shape_of_reasoning_topological.pdf` | Topological trace analysis |
| 17 | Token Complexity of CoT | Lee et al. | 2025 | 77 | `lee2025_token_complexity_cot.pdf` | Per-problem token complexity |
| 18 | Explore Briefly, Then Decide | Liu et al. | 2025 | - | `liu2025_explore_briefly_then_decide.pdf` | Entropy-based early exit |
| 19 | Overclocking LLM Reasoning | Eisenstadt et al. | 2025 | 13 | `eisenstadt2025_overclocking_reasoning.pdf` | Internal progress monitoring |
| 20 | Multi-LogiEval | Patel et al. | 2024 | 40 | `patel2024_multi_logieval.pdf` | Multi-step logic evaluation |
| 21 | Stop Overthinking (Survey) | Sui et al. | 2025 | - | `sui2025_stop_overthinking_survey.pdf` | Efficient reasoning survey |
| 22 | Fractured CoT | Liao et al. | 2025 | 7 | `liao2025_fractured_cot.pdf` | Truncation, accuracy-cost tradeoffs |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets downloaded: 6 (1 gated, not downloaded)

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| GSM8K | HuggingFace `openai/gsm8k` | 8.8K | Math reasoning | `datasets/gsm8k/` | Downloaded |
| MATH-500 | HuggingFace `HuggingFaceH4/MATH-500` | 500 | Competition math | `datasets/math500/` | Downloaded |
| MMLU STEM | HuggingFace `cais/mmlu` | 654 | Multi-subject QA | `datasets/mmlu_stem/` | Downloaded |
| ARC-Challenge | HuggingFace `allenai/ai2_arc` | 2.6K | Science QA | `datasets/arc_challenge/` | Downloaded |
| CommonsenseQA | HuggingFace `tau/commonsense_qa` | 12.1K | Commonsense | `datasets/commonsenseqa/` | Downloaded |
| StrategyQA | HuggingFace `ChilleD/StrategyQA` | 2.3K | Multi-hop QA | `datasets/strategyqa/` | Downloaded |
| GPQA Diamond | HuggingFace `Idavidrein/gpqa` | 448 | Graduate science | `datasets/gpqa/` | Gated (requires auth) |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 6

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| reasoning-step-length | github.com/MingyuJ666/... | Step length manipulation | `code/reasoning-step-length/` |
| l1-length-control | github.com/cmu-l3/l1 | LCPO + controllable length | `code/l1-length-control/` |
| tokenskip | github.com/hemingkx/TokenSkip | CoT compression | `code/tokenskip/` |
| frac-cot | github.com/BaohaoLiao/frac-cot | Fractured sampling | `code/frac-cot/` |
| reasoning-boundary | github.com/LightChen233/... | Reasoning boundary framework | `code/reasoning-boundary/` |
| litecot | github.com/Evanwu1125/LiteCoT | Difficulty-aware pruning | `code/litecot/` |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with 3 diligent searches across complementary query angles
- Supplemented with web search for specific topics (overthinking/underthinking, adaptive budgets)
- Cross-referenced datasets mentioned in papers with HuggingFace availability
- Cloned repositories linked from the most-cited papers

### Selection Criteria
- Papers: Prioritized by relevance (score >= 3), recency (2024-2026), and citation count
- Datasets: Selected to cover an ID-to-OOD gradient from easy math to distant domains
- Code: Focused on repositories with direct experimental utility

### Challenges Encountered
- GPQA dataset is gated on HuggingFace (requires authentication/approval)
- Two paper PDFs were initially mislabeled (fixed by re-downloading correct arXiv IDs)
- Some paper-finder results had truncated JSON requiring manual parsing

### Gaps and Workarounds
- GPQA: Document download instructions; experiment runner can authenticate
- Full MATH dataset (12.5K): Used MATH-500 subset; full set available via `hendrycks/competition_mathematics` if needed
- AIME 2024/2025: Not downloaded as standalone datasets; available through L1 codebase

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary datasets**: GSM8K (ID), MATH-500 (near-OOD), MMLU STEM + ARC + CommonsenseQA + StrategyQA (far-OOD)
2. **Baseline methods**: Vanilla CoT (no control), Budget Forcing (S1), L1-Max (adaptive)
3. **Evaluation metrics**: Accuracy vs token length curves, robustness gap (ID-OOD), length-robustness correlation
4. **Code to adapt/reuse**: L1 codebase (most comprehensive), TokenSkip (compression), reasoning-step-length (manipulation)
5. **Key experimental design**: Sample N=10+ responses per question at varying length budgets, measure accuracy on ID vs OOD, compute correlations between trace length and OOD robustness gap

## Experiment Execution Notes (2026-04-06)
- Implemented real-model evaluation harness in `src/experiment_trace_length.py` using OpenAI Responses API.
- Implemented analysis/statistics pipeline in `src/analyze_trace_length.py`.
- Completed full run with `gpt-4.1`, seed=42, `id_n=18`, `ood_n=6`, `adapt_threshold=0.98`.
- Saved outputs in `results/raw_outputs.jsonl`, `results/metrics.json`, `results/pairwise_tests.json`, and `results/plots/`.
- See `REPORT.md` for full interpretation and scientific conclusions.
