"""Load and standardize all benchmark datasets for the trace-length experiment."""
import json
import random
import os
from datasets import load_from_disk
from pathlib import Path

SEED = 42
BASE = Path(__file__).resolve().parent.parent

def extract_gsm8k_answer(answer_str: str) -> str:
    """Extract numeric answer from GSM8K format '#### NUMBER'."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
    nums = [w.replace(",", "") for w in answer_str.split() if w.replace(",", "").replace(".", "").replace("-", "").isdigit()]
    return nums[-1] if nums else answer_str.strip()

def extract_math_answer(answer_str: str) -> str:
    """Return MATH answer string as-is (boxed LaTeX)."""
    return answer_str.strip()

def load_gsm8k(n=50):
    ds = load_from_disk(str(BASE / "datasets/gsm8k/data"))
    test = ds["test"]
    random.seed(SEED)
    indices = random.sample(range(len(test)), min(n, len(test)))
    items = []
    for i in indices:
        row = test[i]
        items.append({
            "id": f"gsm8k_{i}",
            "dataset": "gsm8k",
            "question": row["question"],
            "gold": extract_gsm8k_answer(row["answer"]),
            "type": "free_response",
            "domain": "math",
            "split": "id",
        })
    return items

def load_math500(n=50):
    ds = load_from_disk(str(BASE / "datasets/math500/data"))
    test = ds["test"]
    random.seed(SEED)
    indices = random.sample(range(len(test)), min(n, len(test)))
    items = []
    for i in indices:
        row = test[i]
        items.append({
            "id": f"math500_{i}",
            "dataset": "math500",
            "question": row["problem"],
            "gold": extract_math_answer(row["answer"]),
            "type": "free_response",
            "domain": "math",
            "split": "ood_near",
            "level": row.get("level", "unknown"),
        })
    return items

def load_arc_challenge(n=50):
    ds = load_from_disk(str(BASE / "datasets/arc_challenge/data"))
    test = ds["test"]
    random.seed(SEED)
    indices = random.sample(range(len(test)), min(n, len(test)))
    items = []
    for i in indices:
        row = test[i]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"  ({l}) {t}" for l, t in zip(labels, texts))
        items.append({
            "id": f"arc_{i}",
            "dataset": "arc_challenge",
            "question": row["question"] + "\n" + choice_str,
            "gold": row["answerKey"],
            "type": "multiple_choice",
            "domain": "science",
            "split": "ood_far",
            "choices": {l: t for l, t in zip(labels, texts)},
        })
    return items

def load_mmlu_stem(n=50):
    # MMLU may be in a special format
    samples_path = BASE / "datasets/mmlu_stem/samples.json"
    data_path = BASE / "datasets/mmlu_stem/data.json"

    if data_path.exists():
        with open(data_path) as f:
            all_data = json.load(f)
        random.seed(SEED)
        all_data = random.sample(all_data, min(n, len(all_data))) if len(all_data) > n else all_data
        items = []
        for idx, row in enumerate(all_data):
            if isinstance(row.get("choices"), str):
                choices = json.loads(row["choices"])
            else:
                choices = row["choices"]
            labels = ["A", "B", "C", "D"]
            choice_str = "\n".join(f"  ({l}) {c}" for l, c in zip(labels, choices))
            gold = labels[row["answer"]] if isinstance(row["answer"], int) else row["answer"]
            items.append({
                "id": f"mmlu_{idx}",
                "dataset": "mmlu_stem",
                "question": row["question"] + "\n" + choice_str,
                "gold": gold,
                "type": "multiple_choice",
                "domain": "stem",
                "split": "ood_far",
            })
        return items

    # Fallback to samples
    with open(samples_path) as f:
        all_data = json.load(f)
    return [{
        "id": f"mmlu_{i}",
        "dataset": "mmlu_stem",
        "question": row["question"],
        "gold": str(row["answer"]),
        "type": "multiple_choice",
        "domain": "stem",
        "split": "ood_far",
    } for i, row in enumerate(all_data)]

def load_commonsenseqa(n=50):
    ds = load_from_disk(str(BASE / "datasets/commonsenseqa/data"))
    # Use validation (test has no labels)
    split = ds["validation"] if "validation" in ds else ds["test"]
    random.seed(SEED)
    indices = random.sample(range(len(split)), min(n, len(split)))
    items = []
    for i in indices:
        row = split[i]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"  ({l}) {t}" for l, t in zip(labels, texts))
        items.append({
            "id": f"csqa_{i}",
            "dataset": "commonsenseqa",
            "question": row["question"] + "\n" + choice_str,
            "gold": row["answerKey"],
            "type": "multiple_choice",
            "domain": "commonsense",
            "split": "ood_far",
        })
    return items

def load_all_datasets(n_per_dataset=50):
    """Load n samples from each dataset."""
    all_items = []
    loaders = [load_gsm8k, load_math500, load_arc_challenge, load_mmlu_stem, load_commonsenseqa]
    for loader in loaders:
        try:
            items = loader(n=n_per_dataset)
            all_items.extend(items)
            print(f"  Loaded {len(items)} from {loader.__name__}")
        except Exception as e:
            print(f"  ERROR loading {loader.__name__}: {e}")
    return all_items

if __name__ == "__main__":
    items = load_all_datasets(n_per_dataset=50)
    print(f"\nTotal: {len(items)} items")
    for ds in set(i["dataset"] for i in items):
        count = sum(1 for i in items if i["dataset"] == ds)
        print(f"  {ds}: {count}")
    # Save
    out = BASE / "results/raw/all_questions.json"
    with open(out, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved to {out}")
