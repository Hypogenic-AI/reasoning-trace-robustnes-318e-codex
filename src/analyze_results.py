"""
Analyze trace-length experiment results.
Produces tables, statistical tests, and visualizations.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load results ────────────────────────────────────────────────────

def load_results():
    results = []
    with open(RESULTS_DIR / "raw/experiment_results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    return results

# ── Aggregate metrics ───────────────────────────────────────────────

def compute_metrics(results):
    """Compute accuracy and token usage per policy × dataset."""
    metrics = defaultdict(lambda: defaultdict(lambda: {
        "correct": [], "tokens": [], "confidence": []
    }))
    for r in results:
        policy = r["policy"]
        ds = r["dataset"]
        split = r["split"]
        metrics[policy][ds]["correct"].append(int(r["correct"]))
        metrics[policy][ds]["tokens"].append(r["total_tokens"])
        if "confidence" in r:
            metrics[policy][ds]["confidence"].append(r["confidence"])

    # Also aggregate by split type
    for r in results:
        policy = r["policy"]
        split = r["split"]
        metrics[policy][f"_split_{split}"]["correct"].append(int(r["correct"]))
        metrics[policy][f"_split_{split}"]["tokens"].append(r["total_tokens"])
        metrics[policy]["_all"]["correct"].append(int(r["correct"]))
        metrics[policy]["_all"]["tokens"].append(r["total_tokens"])

    return metrics

def accuracy_ci(correct_list, alpha=0.05):
    """Compute accuracy with bootstrap CI."""
    n = len(correct_list)
    if n == 0:
        return 0, 0, 0
    acc = np.mean(correct_list)
    # Wilson score interval
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom
    return acc, center - margin, center + margin

# ── Table 1: Accuracy by policy × dataset ───────────────────────────

def print_accuracy_table(metrics):
    policies = ["none", "short", "medium", "long", "adaptive"]
    datasets = ["gsm8k", "math500", "arc_challenge", "mmlu_stem", "commonsenseqa"]
    splits = ["_split_id", "_split_ood_near", "_split_ood_far"]

    print("\n" + "="*90)
    print("TABLE 1: Accuracy by Policy × Dataset")
    print("="*90)
    header = f"{'Policy':<12}" + "".join(f"{ds:<16}" for ds in datasets) + f"{'ID':<10}{'OOD-near':<10}{'OOD-far':<10}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for p in policies:
        row = f"{p:<12}"
        table_data[p] = {}
        for ds in datasets:
            if ds in metrics[p] and metrics[p][ds]["correct"]:
                acc, lo, hi = accuracy_ci(metrics[p][ds]["correct"])
                row += f"{acc:.1%} [{lo:.1%}-{hi:.1%}]  "
                table_data[p][ds] = acc
            else:
                row += f"{'N/A':<16}"
        for sp in splits:
            if sp in metrics[p] and metrics[p][sp]["correct"]:
                acc, _, _ = accuracy_ci(metrics[p][sp]["correct"])
                row += f"{acc:.1%}     "
                table_data[p][sp] = acc
            else:
                row += f"{'N/A':<10}"
        print(row)

    return table_data

# ── Table 2: Token usage ────────────────────────────────────────────

def print_token_table(metrics):
    policies = ["none", "short", "medium", "long", "adaptive"]
    print("\n" + "="*70)
    print("TABLE 2: Average Completion Tokens by Policy")
    print("="*70)
    print(f"{'Policy':<12}{'Mean tokens':<15}{'Median':<12}{'Std':<12}{'Total correct/1k tokens':<20}")
    print("-"*70)

    token_data = {}
    for p in policies:
        all_tokens = metrics[p]["_all"]["tokens"]
        all_correct = metrics[p]["_all"]["correct"]
        if all_tokens:
            mean_t = np.mean(all_tokens)
            med_t = np.median(all_tokens)
            std_t = np.std(all_tokens)
            eff = sum(all_correct) / (sum(all_tokens) / 1000) if sum(all_tokens) > 0 else 0
            print(f"{p:<12}{mean_t:<15.1f}{med_t:<12.1f}{std_t:<12.1f}{eff:<20.2f}")
            token_data[p] = {"mean": mean_t, "median": med_t, "efficiency": eff}
    return token_data

# ── Statistical tests ───────────────────────────────────────────────

def run_statistical_tests(results):
    """Pairwise McNemar tests between policies on shared items."""
    from itertools import combinations

    # Build per-item correctness
    item_correct = defaultdict(dict)
    for r in results:
        item_correct[r["item_id"]][r["policy"]] = int(r["correct"])

    policies = ["none", "short", "medium", "long", "adaptive"]
    print("\n" + "="*70)
    print("TABLE 3: Pairwise McNemar Tests (p-values)")
    print("="*70)

    p_values = []
    comparisons = []

    for p1, p2 in combinations(policies, 2):
        # Get paired outcomes
        both_items = [iid for iid in item_correct if p1 in item_correct[iid] and p2 in item_correct[iid]]
        c1 = [item_correct[iid][p1] for iid in both_items]
        c2 = [item_correct[iid][p2] for iid in both_items]

        # McNemar: count discordant pairs
        b = sum(1 for a, bb in zip(c1, c2) if a == 1 and bb == 0)  # p1 right, p2 wrong
        c = sum(1 for a, bb in zip(c1, c2) if a == 0 and bb == 1)  # p1 wrong, p2 right

        if b + c > 0:
            # McNemar test with continuity correction
            chi2 = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
            p_val = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            p_val = 1.0

        p_values.append(p_val)
        comparisons.append((p1, p2, b, c, p_val))

    # BH correction
    sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
    bh_corrected = [0] * len(p_values)
    m = len(p_values)
    for rank, (idx, pv) in enumerate(sorted_pvals):
        bh_corrected[idx] = min(pv * m / (rank + 1), 1.0)

    print(f"{'Comparison':<25}{'p1>p2':<8}{'p2>p1':<8}{'p-raw':<12}{'p-BH':<12}{'Sig?':<6}")
    print("-"*70)
    for i, (p1, p2, b, c, pv) in enumerate(comparisons):
        sig = "*" if bh_corrected[i] < 0.05 else ""
        print(f"{p1+' vs '+p2:<25}{b:<8}{c:<8}{pv:<12.4f}{bh_corrected[i]:<12.4f}{sig:<6}")

    return comparisons, bh_corrected

# ── Robustness gap analysis ─────────────────────────────────────────

def robustness_gap_analysis(metrics):
    policies = ["none", "short", "medium", "long", "adaptive"]
    print("\n" + "="*70)
    print("TABLE 4: Robustness Gap (ID Accuracy - OOD Accuracy)")
    print("="*70)
    print(f"{'Policy':<12}{'ID Acc':<10}{'OOD-near':<12}{'OOD-far':<12}{'Gap (near)':<12}{'Gap (far)':<12}")
    print("-"*70)

    gap_data = {}
    for p in policies:
        id_acc = np.mean(metrics[p]["_split_id"]["correct"]) if metrics[p]["_split_id"]["correct"] else 0
        near_acc = np.mean(metrics[p]["_split_ood_near"]["correct"]) if metrics[p]["_split_ood_near"]["correct"] else 0
        far_acc = np.mean(metrics[p]["_split_ood_far"]["correct"]) if metrics[p]["_split_ood_far"]["correct"] else 0
        gap_near = id_acc - near_acc
        gap_far = id_acc - far_acc
        print(f"{p:<12}{id_acc:<10.1%}{near_acc:<12.1%}{far_acc:<12.1%}{gap_near:<12.1%}{gap_far:<12.1%}")
        gap_data[p] = {"id": id_acc, "ood_near": near_acc, "ood_far": far_acc,
                       "gap_near": gap_near, "gap_far": gap_far}
    return gap_data

# ── Visualizations ──────────────────────────────────────────────────

def plot_accuracy_by_policy_dataset(metrics):
    """Bar chart: accuracy per policy grouped by dataset."""
    policies = ["none", "short", "medium", "long", "adaptive"]
    datasets = ["gsm8k", "math500", "arc_challenge", "mmlu_stem", "commonsenseqa"]
    labels = ["GSM8K\n(ID)", "MATH-500\n(OOD-near)", "ARC-C\n(OOD-far)", "MMLU-STEM\n(OOD-far)", "CSQA\n(OOD-far)"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.15
    colors = sns.color_palette("Set2", len(policies))

    for i, p in enumerate(policies):
        accs = []
        errs_lo = []
        errs_hi = []
        for ds in datasets:
            if ds in metrics[p] and metrics[p][ds]["correct"]:
                acc, lo, hi = accuracy_ci(metrics[p][ds]["correct"])
                accs.append(acc)
                errs_lo.append(acc - lo)
                errs_hi.append(hi - acc)
            else:
                accs.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
        ax.bar(x + i * width, accs, width, label=p,
               color=colors[i], yerr=[errs_lo, errs_hi], capsize=3)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Trace-Length Policy and Dataset")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Policy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_policy_dataset.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_by_policy_dataset.png'}")

def plot_robustness_gap(gap_data):
    """Robustness gap visualization."""
    policies = ["none", "short", "medium", "long", "adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, gap_key, title in [(axes[0], "gap_near", "Robustness Gap: ID → OOD-near"),
                                (axes[1], "gap_far", "Robustness Gap: ID → OOD-far")]:
        gaps = [gap_data[p][gap_key] for p in policies]
        colors = ['#e74c3c' if g > 0.15 else '#f39c12' if g > 0.05 else '#2ecc71' for g in gaps]
        bars = ax.bar(policies, gaps, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel("Robustness Gap (ID - OOD Accuracy)")
        ax.set_title(title)
        ax.axhline(y=0, color='black', linewidth=0.5)
        for bar, val in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "robustness_gap.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'robustness_gap.png'}")

def plot_accuracy_vs_tokens(metrics):
    """Scatter: accuracy vs avg tokens per policy (efficiency frontier)."""
    policies = ["none", "short", "medium", "long", "adaptive"]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", len(policies))

    for i, p in enumerate(policies):
        all_correct = metrics[p]["_all"]["correct"]
        all_tokens = metrics[p]["_all"]["tokens"]
        if all_correct:
            acc = np.mean(all_correct)
            avg_tok = np.mean(all_tokens)
            ax.scatter(avg_tok, acc, s=200, c=[colors[i]], label=p,
                      edgecolors='black', linewidth=1, zorder=5)
            ax.annotate(p, (avg_tok, acc), textcoords="offset points",
                       xytext=(10, 5), fontsize=10)

    ax.set_xlabel("Average Completion Tokens")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Accuracy-Efficiency Frontier by Policy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_vs_tokens.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_vs_tokens.png'}")

def plot_nonmonotonicity(metrics):
    """Line plot showing accuracy across length policies for each split."""
    policies_ordered = ["none", "short", "medium", "long"]
    x_positions = [0, 1, 2, 3]
    splits = {
        "_split_id": ("In-Distribution (GSM8K)", "o-"),
        "_split_ood_near": ("OOD-Near (MATH-500)", "s--"),
        "_split_ood_far": ("OOD-Far (ARC+MMLU+CSQA)", "^:"),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    for idx, (sp, (label, marker)) in enumerate(splits.items()):
        accs = []
        cis_lo = []
        cis_hi = []
        for p in policies_ordered:
            if sp in metrics[p] and metrics[p][sp]["correct"]:
                acc, lo, hi = accuracy_ci(metrics[p][sp]["correct"])
                accs.append(acc)
                cis_lo.append(acc - lo)
                cis_hi.append(hi - acc)
            else:
                accs.append(0)
                cis_lo.append(0)
                cis_hi.append(0)

        ax.errorbar(x_positions, accs, yerr=[cis_lo, cis_hi],
                    fmt=marker, label=label, color=colors[idx],
                    linewidth=2, markersize=10, capsize=5)

    # Add adaptive as a horizontal band
    for idx, (sp, (label, _)) in enumerate(splits.items()):
        if sp in metrics["adaptive"] and metrics["adaptive"][sp]["correct"]:
            ada_acc = np.mean(metrics["adaptive"][sp]["correct"])
            ax.axhline(y=ada_acc, color=colors[idx], alpha=0.3, linestyle='-.',
                      linewidth=1.5)
            ax.text(3.3, ada_acc, f'adaptive\n({label.split("(")[0].strip()})',
                   fontsize=7, color=colors[idx], va='center')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["None", "Short", "Medium", "Long"])
    ax.set_xlabel("Reasoning Trace Length Policy")
    ax.set_ylabel("Accuracy")
    ax.set_title("Non-Monotonicity Test: Accuracy vs Trace Length by Distribution")
    ax.legend(loc='best')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nonmonotonicity.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'nonmonotonicity.png'}")

def plot_token_distribution(results):
    """Box plot of token usage by policy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    policies = ["none", "short", "medium", "long", "adaptive"]
    data = {p: [r["total_tokens"] for r in results if r["policy"] == p] for p in policies}

    positions = range(len(policies))
    bp = ax.boxplot([data[p] for p in policies], positions=positions,
                    labels=policies, patch_artist=True, widths=0.6)
    colors = sns.color_palette("Set2", len(policies))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Completion Tokens")
    ax.set_title("Token Usage Distribution by Policy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "token_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'token_distribution.png'}")

# ── Calibration analysis for adaptive policy ────────────────────────

def calibration_analysis(results):
    """Analyze confidence calibration for adaptive policy."""
    adaptive_results = [r for r in results if r["policy"] == "adaptive" and "confidence" in r]
    if not adaptive_results:
        print("No adaptive results with confidence data.")
        return {}

    # ECE (Expected Calibration Error)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        in_bin = [r for r in adaptive_results if lo <= r["confidence"] < hi]
        if in_bin:
            bin_accs.append(np.mean([r["correct"] for r in in_bin]))
            bin_confs.append(np.mean([r["confidence"] for r in in_bin]))
            bin_counts.append(len(in_bin))
        else:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)

    total = sum(bin_counts)
    ece = sum(abs(a - c) * n / total for a, c, n in zip(bin_accs, bin_confs, bin_counts)) if total > 0 else 0

    # Brier score
    brier = np.mean([(r["confidence"] - r["correct"])**2 for r in adaptive_results])

    print(f"\n{'='*50}")
    print("Calibration Analysis (Adaptive Policy)")
    print(f"{'='*50}")
    print(f"ECE: {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Mean confidence: {np.mean([r['confidence'] for r in adaptive_results]):.3f}")
    print(f"Mean accuracy: {np.mean([r['correct'] for r in adaptive_results]):.3f}")
    print(f"Long retry rate: {sum(1 for r in adaptive_results if r.get('adaptive_phase') == 'long_retry') / len(adaptive_results):.1%}")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    nonzero = [(c, a) for c, a, n in zip(bin_confs, bin_accs, bin_counts) if n > 0]
    if nonzero:
        confs, accs = zip(*nonzero)
        ax.bar(confs, accs, width=0.08, alpha=0.7, label='Observed', edgecolor='black')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (Adaptive Policy)\nECE={ece:.3f}, Brier={brier:.3f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "calibration.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'calibration.png'}")

    return {"ece": ece, "brier": brier}

# ── Per-dataset accuracy heatmap ────────────────────────────────────

def plot_heatmap(metrics):
    policies = ["none", "short", "medium", "long", "adaptive"]
    datasets = ["gsm8k", "math500", "arc_challenge", "mmlu_stem", "commonsenseqa"]
    labels_ds = ["GSM8K", "MATH-500", "ARC-C", "MMLU-STEM", "CSQA"]

    data = np.zeros((len(policies), len(datasets)))
    for i, p in enumerate(policies):
        for j, ds in enumerate(datasets):
            if ds in metrics[p] and metrics[p][ds]["correct"]:
                data[i, j] = np.mean(metrics[p][ds]["correct"])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(labels_ds)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)

    for i in range(len(policies)):
        for j in range(len(datasets)):
            ax.text(j, i, f'{data[i,j]:.1%}', ha='center', va='center',
                   color='black' if 0.3 < data[i,j] < 0.7 else 'white', fontsize=11)

    plt.colorbar(im, label='Accuracy')
    ax.set_title("Accuracy Heatmap: Policy × Dataset")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_heatmap.png'}")

# ── Main ────────────────────────────────────────────────────────────

def main():
    results = load_results()
    print(f"Loaded {len(results)} results")

    # Check completeness
    from collections import Counter
    policy_counts = Counter(r["policy"] for r in results)
    print(f"Per policy: {dict(policy_counts)}")

    metrics = compute_metrics(results)
    table_data = print_accuracy_table(metrics)
    token_data = print_token_table(metrics)
    comparisons, bh = run_statistical_tests(results)
    gap_data = robustness_gap_analysis(metrics)
    cal_data = calibration_analysis(results)

    # Generate all plots
    print("\nGenerating plots...")
    plot_accuracy_by_policy_dataset(metrics)
    plot_robustness_gap(gap_data)
    plot_accuracy_vs_tokens(metrics)
    plot_nonmonotonicity(metrics)
    plot_token_distribution(results)
    plot_heatmap(metrics)

    # Save summary JSON
    summary = {
        "n_results": len(results),
        "policy_counts": dict(policy_counts),
        "gap_data": gap_data,
        "token_data": token_data,
        "calibration": cal_data,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {RESULTS_DIR / 'summary.json'}")

if __name__ == "__main__":
    main()
