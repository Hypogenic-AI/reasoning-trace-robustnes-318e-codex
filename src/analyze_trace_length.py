#!/usr/bin/env python3
"""Analyze trace-length experiment outputs and generate plots/tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from statsmodels.stats.contingency_tables import mcnemar


def load_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "error" in obj:
                continue
            rows.append(obj)
    if not rows:
        raise RuntimeError("No successful rows found in raw output JSONL.")
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5).clip(0, 1)
    df["total_tokens"] = df["usage"].apply(lambda x: int((x or {}).get("total_tokens", 0)))
    return df


def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(float(a[s].mean() - b[s].mean()))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    # Inputs are paired binary correctness arrays.
    both_correct = int(((a == 1) & (b == 1)).sum())
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    both_wrong = int(((a == 0) & (b == 0)).sum())
    table = [[both_correct, a_only], [b_only, both_wrong]]
    try:
        res = mcnemar(table, exact=False, correction=True)
        return float(res.pvalue)
    except Exception:
        return float("nan")


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=str, default="results/raw_outputs.jsonl")
    parser.add_argument("--metrics-json", type=str, default="results/metrics.json")
    parser.add_argument("--pairwise-json", type=str, default="results/pairwise_tests.json")
    parser.add_argument("--plots-dir", type=str, default="results/plots")
    args = parser.parse_args()

    raw_path = Path(args.raw_jsonl)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(raw_path)

    grouped = []
    for (policy, split_type), sub in df.groupby(["policy", "split_type"]):
        corr = sub["correct"].astype(int).to_numpy()
        conf = sub["confidence"].to_numpy()
        grouped.append(
            {
                "policy": policy,
                "split_type": split_type,
                "n": int(len(sub)),
                "accuracy": float(corr.mean()),
                "avg_total_tokens": float(sub["total_tokens"].mean()),
                "avg_latency_sec": float(sub["latency_sec"].mean()),
                "brier": float(np.mean((conf - corr) ** 2)),
                "ece": ece_score(conf, corr, n_bins=10),
                "confidence_mean": float(conf.mean()),
            }
        )

    metrics_df = pd.DataFrame(grouped).sort_values(["split_type", "policy"])

    # Robustness gap per policy.
    robustness = []
    for policy in sorted(df["policy"].unique()):
        id_acc = df[(df["policy"] == policy) & (df["split_type"] == "id")]["correct"].mean()
        ood_acc = df[(df["policy"] == policy) & (df["split_type"] == "ood")]["correct"].mean()
        robustness.append(
            {
                "policy": policy,
                "id_accuracy": float(id_acc),
                "ood_accuracy": float(ood_acc),
                "robustness_gap": float(id_acc - ood_acc),
            }
        )
    robustness_df = pd.DataFrame(robustness)

    # Pairwise on OOD between adaptive and each fixed policy.
    pairwise = []
    base = df[df["split_type"] == "ood"]
    adaptive = base[base["policy"] == "adaptive"][["qid", "correct", "confidence"]].rename(
        columns={"correct": "adaptive_correct", "confidence": "adaptive_conf"}
    )

    pvals = []
    tmp = []
    for p in ["none", "short", "medium", "long"]:
        comp = base[base["policy"] == p][["qid", "correct", "confidence"]].rename(
            columns={"correct": f"{p}_correct", "confidence": f"{p}_conf"}
        )
        m = adaptive.merge(comp, on="qid", how="inner")
        a = m["adaptive_correct"].astype(int).to_numpy()
        b = m[f"{p}_correct"].astype(int).to_numpy()

        diff_mean, diff_lo, diff_hi = bootstrap_diff(a, b, n_boot=2000, seed=42)
        pval = mcnemar_p(a, b)
        pvals.append(pval)
        row = {
            "comparison": f"adaptive_vs_{p}",
            "n": int(len(m)),
            "accuracy_diff": float(a.mean() - b.mean()),
            "bootstrap_mean": diff_mean,
            "bootstrap_ci95_low": diff_lo,
            "bootstrap_ci95_high": diff_hi,
            "mcnemar_p": pval,
        }
        tmp.append(row)

    adj = benjamini_hochberg(pvals)
    for r, q in zip(tmp, adj):
        r["mcnemar_q_bh"] = q
        pairwise.append(r)

    # Non-monotonicity proxy: Kendall tau between length rank and OOD accuracy.
    length_rank = {"none": 0, "short": 1, "medium": 2, "long": 3}
    fixed = robustness_df[robustness_df["policy"].isin(length_rank.keys())].copy()
    fixed["rank"] = fixed["policy"].map(length_rank)
    tau, tau_p = kendalltau(fixed["rank"].to_numpy(), fixed["ood_accuracy"].to_numpy())

    # Save metrics
    payload = {
        "by_policy_split": metrics_df.to_dict(orient="records"),
        "robustness": robustness_df.to_dict(orient="records"),
        "non_monotonicity_test": {
            "method": "kendall_tau",
            "tau": float(tau) if tau == tau else None,
            "p_value": float(tau_p) if tau_p == tau_p else None,
            "note": "tau near 1 indicates monotonic increase, near -1 monotonic decrease; near 0 suggests non-monotonic/no ordinal trend.",
        },
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(args.pairwise_json, "w", encoding="utf-8") as f:
        json.dump(pairwise, f, indent=2)

    # Plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=robustness_df, x="policy", y="ood_accuracy", palette="viridis")
    plt.title("OOD Accuracy by Trace Policy")
    plt.ylabel("Accuracy")
    plt.xlabel("Policy")
    plt.tight_layout()
    plt.savefig(plots_dir / "ood_accuracy_by_policy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=robustness_df, x="policy", y="robustness_gap", palette="magma")
    plt.title("Robustness Gap (ID - OOD)")
    plt.ylabel("Gap")
    plt.xlabel("Policy")
    plt.tight_layout()
    plt.savefig(plots_dir / "robustness_gap_by_policy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="policy", y="avg_total_tokens", hue="split_type")
    plt.title("Average Total Tokens by Policy and Split")
    plt.ylabel("Tokens")
    plt.xlabel("Policy")
    plt.tight_layout()
    plt.savefig(plots_dir / "token_usage_by_policy_split.png", dpi=180)
    plt.close()

    # Reliability diagram-like scatter per policy
    rel_rows = []
    for policy, sub in df.groupby("policy"):
        rel_rows.append({
            "policy": policy,
            "accuracy": float(sub["correct"].mean()),
            "mean_confidence": float(sub["confidence"].mean()),
        })
    rel_df = pd.DataFrame(rel_rows)
    plt.figure(figsize=(6, 6))
    plt.scatter(rel_df["mean_confidence"], rel_df["accuracy"], s=80)
    for _, r in rel_df.iterrows():
        plt.text(r["mean_confidence"] + 0.005, r["accuracy"] + 0.005, r["policy"], fontsize=9)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.title("Confidence vs Accuracy by Policy")
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_vs_accuracy.png", dpi=180)
    plt.close()

    print("Saved:")
    print(f"- {args.metrics_json}")
    print(f"- {args.pairwise_json}")
    print(f"- {plots_dir}")


if __name__ == "__main__":
    main()
