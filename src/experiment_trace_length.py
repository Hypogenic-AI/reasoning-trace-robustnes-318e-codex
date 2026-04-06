#!/usr/bin/env python3
"""Run reasoning trace-length experiments with real LLM API calls."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, load_from_disk
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("\\n", " ")
    s = re.sub(r"\\s+", " ", s)
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def extract_first_number(s: str) -> str | None:
    m = re.search(r"-?\\d+(?:\\.\\d+)?", s.replace(",", ""))
    return m.group(0) if m else None


def extract_choice_letter(s: str) -> str | None:
    s = s.strip().upper()
    m = re.search(r"\b([A-F])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\(([A-F])\)", s)
    return m.group(1) if m else None


def parse_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        val = answer.split("####")[-1].strip()
    else:
        val = answer.strip()
    num = extract_first_number(val)
    return num if num is not None else normalize_text(val)


@dataclass
class Example:
    dataset: str
    split_type: str
    qid: str
    question: str
    gold: str
    answer_type: str  # numeric | choice | text


POLICIES: dict[str, dict[str, Any]] = {
    "none": {
        "instruction": "Provide only the final answer. No reasoning.",
        "max_output_tokens": 80,
    },
    "short": {
        "instruction": "Use at most 2 concise reasoning steps, then final answer.",
        "max_output_tokens": 180,
    },
    "medium": {
        "instruction": "Use about 4-6 reasoning steps, then final answer.",
        "max_output_tokens": 320,
    },
    "long": {
        "instruction": "Use detailed reasoning with at least 8 explicit steps, then final answer.",
        "max_output_tokens": 700,
    },
}


def build_eval_set(base_dir: Path, id_n: int, ood_n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []

    gsm = load_from_disk(str(base_dir / "gsm8k_main"))["test"]
    gsm_idx = list(range(len(gsm)))
    rng.shuffle(gsm_idx)
    for i in gsm_idx[:id_n]:
        r = gsm[i]
        rows.append(
            Example(
                dataset="gsm8k",
                split_type="id",
                qid=f"gsm8k_test_{i}",
                question=r["question"],
                gold=parse_gsm8k_gold(r["answer"]),
                answer_type="numeric",
            )
        )

    arc = load_from_disk(str(base_dir / "ai2_arc_challenge"))["test"]
    arc_idx = list(range(len(arc)))
    rng.shuffle(arc_idx)
    for i in arc_idx[:ood_n]:
        r = arc[i]
        choices = r["choices"]
        choice_lines = []
        for label, text in zip(choices["label"], choices["text"]):
            choice_lines.append(f"({label}) {text}")
        q = f"{r['question']}\\nOptions:\\n" + "\\n".join(choice_lines)
        rows.append(
            Example(
                dataset="arc_challenge",
                split_type="ood",
                qid=f"arc_test_{i}",
                question=q,
                gold=r["answerKey"].strip().upper(),
                answer_type="choice",
            )
        )

    for bbh_name in [
        "bbh_date_understanding",
        "bbh_logical_deduction_three_objects",
        "bbh_multistep_arithmetic_two",
    ]:
        ds = load_from_disk(str(base_dir / bbh_name))["test"]
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:ood_n]:
            r = ds[i]
            target = r["target"]
            ans_type = "choice" if re.search(r"\([A-F]\)", target.upper()) else "numeric"
            gold = extract_choice_letter(target) if ans_type == "choice" else (extract_first_number(target) or normalize_text(target))
            rows.append(
                Example(
                    dataset=bbh_name,
                    split_type="ood",
                    qid=f"{bbh_name}_test_{i}",
                    question=r["input"],
                    gold=gold or normalize_text(target),
                    answer_type=ans_type,
                )
            )

    math500 = load_from_disk(str(base_dir / "math500"))["test"]
    m_idx = list(range(len(math500)))
    rng.shuffle(m_idx)
    for i in m_idx[:ood_n]:
        r = math500[i]
        rows.append(
            Example(
                dataset="math500",
                split_type="ood",
                qid=f"math500_test_{i}",
                question=r["problem"],
                gold=normalize_text(str(r["answer"])),
                answer_type="text",
            )
        )

    return rows


def parse_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def is_correct(pred: str, gold: str, answer_type: str) -> bool:
    if answer_type == "choice":
        p = extract_choice_letter(pred or "")
        return (p or "") == (gold or "")
    if answer_type == "numeric":
        p = extract_first_number(pred or "")
        g = extract_first_number(gold or "")
        return (p or "") == (g or "")

    # text/latex fallback
    return normalize_text(pred or "") == normalize_text(gold or "")


def policy_prompt(question: str, policy: str) -> str:
    base = (
        "You are solving a reasoning problem. Return strict JSON with keys: "
        "reasoning (string), final_answer (string), confidence (float between 0 and 1), "
        "uncertainty_note (short string). "
        "Do not include any keys besides these four."
    )
    if policy == "adaptive_short_probe":
        inst = "Use at most 2 concise reasoning steps and provide your best answer."
    else:
        inst = POLICIES[policy]["instruction"]
    return f"{base}\\n{inst}\\n\\nProblem:\\n{question}"


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_model(client: OpenAI, model: str, prompt: str, max_output_tokens: int, temperature: float) -> tuple[str, dict[str, int]]:
    # responses API for current OpenAI SDK.
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = resp.output_text
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0,
        "output_tokens": getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) if getattr(resp, "usage", None) else 0,
    }
    return text, usage


def run_policy(
    client: OpenAI,
    model: str,
    ex: Example,
    policy: str,
    temperature: float,
    adapt_threshold: float,
) -> dict[str, Any]:
    start = time.time()

    if policy != "adaptive":
        prompt = policy_prompt(ex.question, policy)
        raw, usage = call_model(
            client=client,
            model=model,
            prompt=prompt,
            max_output_tokens=POLICIES[policy]["max_output_tokens"],
            temperature=temperature,
        )
        payload = parse_json_from_text(raw)
        final_answer = str(payload.get("final_answer", "")).strip()
        confidence = float(payload.get("confidence", 0.5)) if str(payload.get("confidence", "")).strip() else 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {
            "policy": policy,
            "raw_text": raw,
            "reasoning": str(payload.get("reasoning", "")),
            "final_answer": final_answer,
            "confidence": confidence,
            "uncertainty_note": str(payload.get("uncertainty_note", "")),
            "usage": usage,
            "latency_sec": time.time() - start,
            "adaptive_second_pass": False,
            "adaptive_first_confidence": confidence,
        }

    # Adaptive: short probe, then escalate if low confidence.
    prompt1 = policy_prompt(ex.question, "adaptive_short_probe")
    raw1, usage1 = call_model(
        client=client,
        model=model,
        prompt=prompt1,
        max_output_tokens=POLICIES["short"]["max_output_tokens"],
        temperature=temperature,
    )
    payload1 = parse_json_from_text(raw1)
    conf1 = float(payload1.get("confidence", 0.5)) if str(payload1.get("confidence", "")).strip() else 0.5
    conf1 = max(0.0, min(1.0, conf1))

    if conf1 >= adapt_threshold:
        final_answer = str(payload1.get("final_answer", "")).strip()
        return {
            "policy": policy,
            "raw_text": raw1,
            "reasoning": str(payload1.get("reasoning", "")),
            "final_answer": final_answer,
            "confidence": conf1,
            "uncertainty_note": str(payload1.get("uncertainty_note", "")),
            "usage": usage1,
            "latency_sec": time.time() - start,
            "adaptive_second_pass": False,
            "adaptive_first_confidence": conf1,
        }

    prompt2 = policy_prompt(ex.question, "long")
    raw2, usage2 = call_model(
        client=client,
        model=model,
        prompt=prompt2,
        max_output_tokens=POLICIES["long"]["max_output_tokens"],
        temperature=temperature,
    )
    payload2 = parse_json_from_text(raw2)
    conf2 = float(payload2.get("confidence", 0.5)) if str(payload2.get("confidence", "")).strip() else 0.5
    conf2 = max(0.0, min(1.0, conf2))

    usage = {
        "input_tokens": usage1["input_tokens"] + usage2["input_tokens"],
        "output_tokens": usage1["output_tokens"] + usage2["output_tokens"],
        "total_tokens": usage1["total_tokens"] + usage2["total_tokens"],
    }
    final_answer = str(payload2.get("final_answer", "")).strip()
    return {
        "policy": policy,
        "raw_text": raw2,
        "reasoning": str(payload2.get("reasoning", "")),
        "final_answer": final_answer,
        "confidence": conf2,
        "uncertainty_note": str(payload2.get("uncertainty_note", "")),
        "usage": usage,
        "latency_sec": time.time() - start,
        "adaptive_second_pass": True,
        "adaptive_first_confidence": conf1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument("--out-jsonl", type=str, default="results/raw_outputs.jsonl")
    parser.add_argument("--summary-json", type=str, default="results/run_summary.json")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4.1"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-n", type=int, default=20)
    parser.add_argument("--ood-n", type=int, default=8)
    parser.add_argument("--adapt-threshold", type=float, default=0.72)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    set_seed(args.seed)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    examples = build_eval_set(Path(args.datasets_dir), id_n=args.id_n, ood_n=args.ood_n, seed=args.seed)
    policies = ["none", "short", "medium", "long", "adaptive"]

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "id_n": args.id_n,
        "ood_n_per_dataset": args.ood_n,
        "n_examples": len(examples),
        "policies": policies,
        "adapt_threshold": args.adapt_threshold,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Examples"):
            for policy in policies:
                try:
                    result = run_policy(
                        client=client,
                        model=args.model,
                        ex=ex,
                        policy=policy,
                        temperature=args.temperature,
                        adapt_threshold=args.adapt_threshold,
                    )
                    pred = result["final_answer"]
                    correct = is_correct(pred=pred, gold=ex.gold, answer_type=ex.answer_type)

                    row = {
                        "dataset": ex.dataset,
                        "split_type": ex.split_type,
                        "qid": ex.qid,
                        "question": ex.question,
                        "gold": ex.gold,
                        "answer_type": ex.answer_type,
                        "model": args.model,
                        "policy": policy,
                        "prediction": pred,
                        "correct": bool(correct),
                        "confidence": float(result["confidence"]),
                        "uncertainty_note": result["uncertainty_note"],
                        "reasoning": result["reasoning"],
                        "usage": result["usage"],
                        "latency_sec": result["latency_sec"],
                        "adaptive_second_pass": result["adaptive_second_pass"],
                        "adaptive_first_confidence": result["adaptive_first_confidence"],
                        "raw_text": result["raw_text"],
                    }
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    f.flush()
                except Exception as exc:
                    err = {
                        "dataset": ex.dataset,
                        "split_type": ex.split_type,
                        "qid": ex.qid,
                        "model": args.model,
                        "policy": policy,
                        "error": str(exc),
                    }
                    f.write(json.dumps(err, ensure_ascii=True) + "\n")
                    f.flush()

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as sf:
        json.dump(run_meta, sf, indent=2)

    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
