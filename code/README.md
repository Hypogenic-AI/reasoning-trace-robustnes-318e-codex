# Cloned Repositories

## Repo 1: reasoning-step-length
- **URL**: https://github.com/MingyuJ666/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models
- **Paper**: "The Impact of Reasoning Step Length on Large Language Models" (Jin et al., 2024)
- **Purpose**: Code for expanding/compressing reasoning steps in CoT demonstrations
- **Location**: `code/reasoning-step-length/`
- **Key files**: Scripts for step length manipulation experiments
- **Notes**: Foundational implementation for studying step length effects

## Repo 2: l1-length-control
- **URL**: https://github.com/cmu-l3/l1
- **Paper**: "L1: Controlling How Long A Reasoning Model Thinks With RL" (Aggarwal & Welleck, 2025)
- **Purpose**: LCPO training code, L1 model weights, evaluation scripts
- **Location**: `code/l1-length-control/`
- **Key files**: Training scripts, evaluation configs, model links
- **Notes**: Most relevant codebase - provides controllable length reasoning models

## Repo 3: tokenskip
- **URL**: https://github.com/hemingkx/TokenSkip
- **Paper**: "TokenSkip: Controllable Chain-of-Thought Compression in LLMs" (Xia et al., 2025)
- **Purpose**: Token-level CoT compression with importance scoring
- **Location**: `code/tokenskip/`
- **Key files**: Compression scripts, evaluation code
- **Notes**: Useful as compression baseline for experiments

## Repo 4: frac-cot
- **URL**: https://github.com/BaohaoLiao/frac-cot
- **Paper**: "Fractured Chain-of-Thought Reasoning" (Liao et al., 2025)
- **Purpose**: Fractured sampling: interpolating between full CoT and solution-only
- **Location**: `code/frac-cot/`
- **Key files**: Sampling code, benchmark evaluation scripts
- **Notes**: Useful for studying accuracy-cost tradeoffs across reasoning depths

## Repo 5: reasoning-boundary
- **URL**: https://github.com/LightChen233/reasoning-boundary
- **Paper**: "Unlocking the Capabilities of Thought: A Reasoning Boundary Framework" (Chen et al., 2024)
- **Purpose**: Quantitative framework for CoT capabilities with combination laws
- **Location**: `code/reasoning-boundary/`
- **Key files**: Framework code, 27-model evaluation scripts
- **Notes**: Provides metrics for quantifying reasoning boundaries

## Repo 6: litecot
- **URL**: https://github.com/Evanwu1125/LiteCoT
- **Paper**: "Concise Reasoning, Big Gains" (Wu et al., 2025)
- **Purpose**: Difficulty-aware CoT pruning, LiteCoT distilled dataset
- **Location**: `code/litecot/`
- **Key files**: DAP pipeline, distillation scripts
- **Notes**: 100K concise reasoning examples, Liter model family
