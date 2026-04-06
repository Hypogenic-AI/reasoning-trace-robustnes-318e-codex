# Downloaded Datasets

This directory contains datasets for the research project on reasoning trace length and robustness.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: GSM8K

### Overview
- **Source**: HuggingFace `openai/gsm8k`
- **Size**: Train: 7,473 / Test: 1,319 problems
- **Format**: HuggingFace Dataset
- **Task**: Grade-school math word problems (2-8 step solutions)
- **License**: MIT
- **Role**: In-distribution (ID) math reasoning benchmark

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k/data")
```

---

## Dataset 2: MATH-500

### Overview
- **Source**: HuggingFace `HuggingFaceH4/MATH-500`
- **Size**: 500 test problems
- **Format**: HuggingFace Dataset
- **Task**: Competition-level mathematics (algebra, geometry, number theory, etc.)
- **Difficulty levels**: Level 1-5
- **Role**: Hard math benchmark, OOD relative to GSM8K

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/MATH-500")
dataset.save_to_disk("datasets/math500/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/math500/data")
```

---

## Dataset 3: MMLU STEM

### Overview
- **Source**: HuggingFace `cais/mmlu` (selected STEM subjects)
- **Size**: 654 problems (abstract_algebra, college_math, college_physics, hs_math)
- **Format**: JSON
- **Task**: Multiple-choice questions across STEM subjects
- **Role**: Cross-domain OOD evaluation (beyond math)

### Download Instructions
```python
from datasets import load_dataset
subjects = ["abstract_algebra", "college_mathematics", "college_physics", "high_school_mathematics"]
all_data = []
for subj in subjects:
    ds = load_dataset("cais/mmlu", subj)
    for split in ds:
        for item in ds[split]:
            all_data.append({"subject": subj, "question": item["question"],
                           "choices": item["choices"], "answer": item["answer"]})
import json
with open("datasets/mmlu_stem/data.json", 'w') as f:
    json.dump(all_data, f)
```

### Loading
```python
import json
with open("datasets/mmlu_stem/data.json") as f:
    data = json.load(f)
```

---

## Dataset 4: ARC-Challenge

### Overview
- **Source**: HuggingFace `allenai/ai2_arc` (ARC-Challenge)
- **Size**: Train: 1,119 / Test: 1,172 / Val: 299
- **Format**: HuggingFace Dataset
- **Task**: Grade-school science questions (multiple choice)
- **Role**: OOD science reasoning evaluation

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset.save_to_disk("datasets/arc_challenge/data")
```

---

## Dataset 5: CommonsenseQA

### Overview
- **Source**: HuggingFace `tau/commonsense_qa`
- **Size**: Train: 9,741 / Val: 1,221 / Test: 1,140
- **Format**: HuggingFace Dataset
- **Task**: Commonsense reasoning (multiple choice)
- **Role**: OOD commonsense evaluation

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("tau/commonsense_qa")
dataset.save_to_disk("datasets/commonsenseqa/data")
```

---

## Dataset 6: StrategyQA

### Overview
- **Source**: HuggingFace `ChilleD/StrategyQA`
- **Size**: Train: 1,603 / Test: 687
- **Format**: HuggingFace Dataset
- **Task**: Multi-hop yes/no reasoning questions
- **Role**: OOD multi-hop reasoning evaluation

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("ChilleD/StrategyQA")
dataset.save_to_disk("datasets/strategyqa/data")
```

---

## Dataset 7: GPQA Diamond (NOT DOWNLOADED - gated)

### Overview
- **Source**: HuggingFace `Idavidrein/gpqa` (gated, requires authentication)
- **Size**: 448 graduate-level science questions
- **Task**: Graduate-level science reasoning (biology, chemistry, physics)
- **Role**: Very hard OOD reasoning benchmark

### Download Instructions
Requires HuggingFace authentication:
```python
from datasets import load_dataset
# Must be logged in: huggingface-cli login
dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
dataset.save_to_disk("datasets/gpqa/data")
```

---

## Experimental Design Notes

**In-Distribution (ID)**: GSM8K serves as the primary ID benchmark (simple math reasoning).

**Out-of-Distribution (OOD)** evaluation uses progressively distant domains:
1. MATH-500 (harder math, same domain)
2. MMLU STEM (broader STEM, different format)
3. ARC-Challenge (science reasoning)
4. CommonsenseQA (commonsense, very different)
5. StrategyQA (multi-hop, different reasoning type)

This gradient of distribution shift enables studying how trace length-robustness relationships change with increasing OOD distance.
