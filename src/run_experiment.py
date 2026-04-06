"""
Run trace-length experiment: 5 policies × 250 questions using GPT-4.1.

Policies:
  none   - Direct answer, no reasoning
  short  - ≤2 brief reasoning steps
  medium - 4-6 reasoning steps
  long   - ≥8 detailed reasoning steps
  adaptive - Short first; if low confidence, retry with long
"""
import json
import os
import re
import time
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

SEED = 42
random.seed(SEED)
BASE = Path(__file__).resolve().parent.parent
MODEL = "gpt-4.1-nano"  # cost-effective for 1250+ calls; upgrade if budget allows

client = OpenAI()

# ── Prompt templates ────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a precise problem solver. Always end your response with your final answer on a new line in the format: ANSWER: <your answer>"

POLICY_INSTRUCTIONS = {
    "none": (
        "Answer the following question directly with NO reasoning or explanation. "
        "Just provide the answer immediately.\n\n{question}\n\n"
        "ANSWER:"
    ),
    "short": (
        "Answer the following question. Provide at most 1-2 brief reasoning sentences "
        "before your answer.\n\n{question}\n\n"
        "Think briefly, then give your answer."
    ),
    "medium": (
        "Answer the following question. Show your reasoning in 4-6 clear steps "
        "before giving your final answer.\n\n{question}\n\n"
        "Think step by step."
    ),
    "long": (
        "Answer the following question. Provide very detailed, thorough reasoning "
        "with at least 8 steps. Consider multiple angles, check your work, and "
        "verify your answer before finalizing.\n\n{question}\n\n"
        "Think very carefully and thoroughly, step by step."
    ),
}

CONFIDENCE_PROMPT = (
    "Answer the following question. Provide 1-2 brief reasoning sentences, "
    "then give your answer AND your confidence (0-100%).\n\n{question}\n\n"
    "Format: [reasoning]\nANSWER: <answer>\nCONFIDENCE: <0-100>"
)

LONG_RETRY_PROMPT = (
    "You previously attempted this question but were not confident. "
    "Now think very carefully with detailed step-by-step reasoning (8+ steps). "
    "Check your work thoroughly.\n\n{question}\n\n"
    "Think very carefully and thoroughly, step by step."
)

# ── Max tokens per policy ───────────────────────────────────────────

MAX_TOKENS = {
    "none": 100,
    "short": 300,
    "medium": 800,
    "long": 2000,
    "adaptive_short": 400,
    "adaptive_long": 2000,
}

# ── Answer extraction ───────────────────────────────────────────────

def extract_answer(text: str, qtype: str) -> str:
    """Extract the final answer from model output."""
    text = text.strip()

    # Try ANSWER: pattern first
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
    else:
        # Last line fallback
        ans = text.strip().split('\n')[-1].strip()

    # Clean up
    ans = re.sub(r'^["\']|["\']$', '', ans)
    ans = ans.strip('.')

    if qtype == "multiple_choice":
        # Extract letter
        m = re.search(r'\b([A-E])\b', ans)
        if m:
            return m.group(1)
        return ans.upper()[:1] if ans else ""

    if qtype == "free_response":
        # Extract number for math
        # Try boxed first
        m = re.search(r'\\boxed\{([^}]+)\}', ans)
        if m:
            return m.group(1).strip()
        # Try plain number
        m = re.search(r'(-?\d[\d,]*\.?\d*)', ans.replace(",", ""))
        if m:
            return m.group(1)
        return ans

    return ans

def extract_confidence(text: str) -> float:
    """Extract confidence from model output."""
    m = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r'(\d+)\s*%', text)
    if m:
        return float(m.group(1)) / 100.0
    return 0.5  # default

def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip().lower()
    ans = re.sub(r'[,$\\{}\s]', '', ans)
    # Remove leading zeros for numbers
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans

def check_correct(predicted: str, gold: str, qtype: str) -> bool:
    """Check if predicted answer matches gold."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    # For MC, just compare first letter
    if qtype == "multiple_choice":
        return pred_norm[:1] == gold_norm[:1]
    # Numeric approximate match
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except ValueError:
        return pred_norm == gold_norm

# ── API call ────────────────────────────────────────────────────────

def call_model(prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    """Call OpenAI API and return response + usage info."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            return {"text": msg, "usage": usage, "model": resp.model}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "error": str(e)}

# ── Run single item ─────────────────────────────────────────────────

def run_item_policy(item: dict, policy: str) -> dict:
    """Run a single question under a given policy."""
    q = item["question"]
    qtype = item["type"]

    if policy == "adaptive":
        # Phase 1: short attempt with confidence
        prompt1 = CONFIDENCE_PROMPT.format(question=q)
        resp1 = call_model(prompt1, MAX_TOKENS["adaptive_short"])
        conf = extract_confidence(resp1["text"])
        ans1 = extract_answer(resp1["text"], qtype)

        if conf >= 0.7:
            # Confident enough, use short answer
            return {
                "item_id": item["id"],
                "policy": "adaptive",
                "adaptive_phase": "short_only",
                "answer": ans1,
                "confidence": conf,
                "correct": check_correct(ans1, item["gold"], qtype),
                "text": resp1["text"],
                "usage": resp1["usage"],
                "total_tokens": resp1["usage"]["completion_tokens"],
            }
        else:
            # Low confidence, retry with long reasoning
            prompt2 = LONG_RETRY_PROMPT.format(question=q)
            resp2 = call_model(prompt2, MAX_TOKENS["adaptive_long"])
            ans2 = extract_answer(resp2["text"], qtype)
            total_comp = resp1["usage"]["completion_tokens"] + resp2["usage"]["completion_tokens"]
            total_prompt = resp1["usage"]["prompt_tokens"] + resp2["usage"]["prompt_tokens"]
            return {
                "item_id": item["id"],
                "policy": "adaptive",
                "adaptive_phase": "long_retry",
                "answer": ans2,
                "confidence": conf,
                "correct": check_correct(ans2, item["gold"], qtype),
                "text": resp2["text"],
                "text_short": resp1["text"],
                "usage": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_comp,
                    "total_tokens": total_prompt + total_comp,
                },
                "total_tokens": total_comp,
            }
    else:
        # Fixed policy
        prompt = POLICY_INSTRUCTIONS[policy].format(question=q)
        resp = call_model(prompt, MAX_TOKENS[policy])
        ans = extract_answer(resp["text"], qtype)
        return {
            "item_id": item["id"],
            "policy": policy,
            "answer": ans,
            "correct": check_correct(ans, item["gold"], qtype),
            "text": resp["text"],
            "usage": resp["usage"],
            "total_tokens": resp["usage"]["completion_tokens"],
        }

# ── Main ────────────────────────────────────────────────────────────

def main():
    # Load questions
    with open(BASE / "results/raw/all_questions.json") as f:
        items = json.load(f)

    policies = ["none", "short", "medium", "long", "adaptive"]
    all_results = []
    output_path = BASE / "results/raw/experiment_results.jsonl"

    # Check for existing results to resume
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["item_id"], r["policy"]))
        print(f"Resuming: {len(existing)} results already collected")

    total = len(items) * len(policies)
    remaining = total - len(existing)
    print(f"Total calls needed: {total}, remaining: {remaining}")

    with open(output_path, "a") as fout:
        for policy in policies:
            print(f"\n=== Policy: {policy} ===")
            for item in tqdm(items, desc=policy):
                key = (item["id"], policy)
                if key in existing:
                    continue

                result = run_item_policy(item, policy)
                # Add metadata
                result["dataset"] = item["dataset"]
                result["domain"] = item["domain"]
                result["split"] = item["split"]
                result["gold"] = item["gold"]
                result["qtype"] = item["type"]

                fout.write(json.dumps(result) + "\n")
                fout.flush()

    print(f"\nDone! Results saved to {output_path}")

if __name__ == "__main__":
    main()
