"""
Confirmatory experiment with gpt-4.1-mini on a smaller subset (20 per dataset).
Tests whether non-monotonicity patterns hold with a stronger model.
"""
import json
import os
import re
import time
import random
from pathlib import Path
from openai import OpenAI

SEED = 42
random.seed(SEED)
BASE = Path(__file__).resolve().parent.parent
MODEL = "gpt-4.1-mini"

client = OpenAI()

# Reuse the same prompts
SYSTEM_PROMPT = "You are a precise problem solver. Always end your response with your final answer on a new line in the format: ANSWER: <your answer>"

POLICY_INSTRUCTIONS = {
    "none": "Answer the following question directly with NO reasoning or explanation. Just provide the answer immediately.\n\n{question}\n\nANSWER:",
    "short": "Answer the following question. Provide at most 1-2 brief reasoning sentences before your answer.\n\n{question}\n\nThink briefly, then give your answer.",
    "medium": "Answer the following question. Show your reasoning in 4-6 clear steps before giving your final answer.\n\n{question}\n\nThink step by step.",
    "long": "Answer the following question. Provide very detailed, thorough reasoning with at least 8 steps. Consider multiple angles, check your work, and verify your answer before finalizing.\n\n{question}\n\nThink very carefully and thoroughly, step by step.",
}

MAX_TOKENS = {"none": 100, "short": 300, "medium": 800, "long": 2000}

def extract_answer(text, qtype):
    text = text.strip()
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    ans = match.group(1).strip() if match else text.strip().split('\n')[-1].strip()
    ans = re.sub(r'^["\']|["\']$', '', ans).strip('.')
    if qtype == "multiple_choice":
        m = re.search(r'\b([A-E])\b', ans)
        return m.group(1) if m else ans.upper()[:1]
    m = re.search(r'\\boxed\{([^}]+)\}', ans)
    if m: return m.group(1).strip()
    m = re.search(r'(-?\d[\d,]*\.?\d*)', ans.replace(",", ""))
    if m: return m.group(1)
    return ans

def normalize_answer(ans):
    ans = re.sub(r'[,$\\{}\s]', '', ans.strip().lower())
    try:
        val = float(ans)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return ans

def check_correct(pred, gold, qtype):
    pn, gn = normalize_answer(pred), normalize_answer(gold)
    if pn == gn: return True
    if qtype == "multiple_choice": return pn[:1] == gn[:1]
    try: return abs(float(pn) - float(gn)) < 1e-6
    except: return False

def call_model(prompt, max_tokens):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ], max_tokens=max_tokens, temperature=0.0)
            return {
                "text": resp.choices[0].message.content or "",
                "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                          "completion_tokens": resp.usage.completion_tokens,
                          "total_tokens": resp.usage.total_tokens}
            }
        except Exception as e:
            if attempt < 2: time.sleep(2**attempt)
            else: return {"text": "", "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}, "error": str(e)}

def main():
    with open(BASE / "results/raw/all_questions.json") as f:
        all_items = json.load(f)

    # Take first 20 per dataset
    from collections import defaultdict
    by_ds = defaultdict(list)
    for item in all_items:
        by_ds[item["dataset"]].append(item)

    items = []
    for ds, ds_items in by_ds.items():
        items.extend(ds_items[:20])

    print(f"Running {len(items)} items × 4 policies = {len(items)*4} calls with {MODEL}")

    results = []
    policies = ["none", "short", "medium", "long"]

    for policy in policies:
        print(f"\n=== Policy: {policy} ===")
        for item in items:
            prompt = POLICY_INSTRUCTIONS[policy].format(question=item["question"])
            resp = call_model(prompt, MAX_TOKENS[policy])
            ans = extract_answer(resp["text"], item["type"])
            correct = check_correct(ans, item["gold"], item["type"])
            results.append({
                "item_id": item["id"], "policy": policy, "answer": ans,
                "correct": correct, "dataset": item["dataset"], "split": item["split"],
                "gold": item["gold"], "qtype": item["type"],
                "total_tokens": resp["usage"]["completion_tokens"],
            })
            print("." if correct else "x", end="", flush=True)
        print()

    # Save
    out = BASE / "results/raw/confirmatory_results.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Quick summary
    from collections import Counter
    for policy in policies:
        pr = [r for r in results if r["policy"] == policy]
        acc = sum(r["correct"] for r in pr) / len(pr)
        avg_tok = sum(r["total_tokens"] for r in pr) / len(pr)
        print(f"{policy}: acc={acc:.1%}, avg_tokens={avg_tok:.0f}")
        # By split
        for sp in ["id", "ood_near", "ood_far"]:
            spr = [r for r in pr if r["split"] == sp]
            if spr:
                sacc = sum(r["correct"] for r in spr) / len(spr)
                print(f"  {sp}: {sacc:.1%} (n={len(spr)})")

if __name__ == "__main__":
    main()
