# -*- coding: utf-8 -*-
"""Create publication-ready experiment figures from frozen JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compositional_kge import OUT

FIG = Path(__file__).resolve().parent / "figures"
COLORS = {"Hybrid PairComp": "#B2182B", "Parameter control": "#2166AC", "DeepSets": "#4D9221", "Pair only": "#762A83"}


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def overall_and_size():
    raw = read("hybrid_confirmatory_detailed_analysis.json")
    mapping = [("Hybrid PairComp", "hybrid"), ("Parameter control", "parameter_control"), ("DeepSets", "deepsets"), ("Pair only", "pair_only")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    metrics = ["mrr", "hits1", "hits3", "hits10"]
    x = np.arange(len(metrics)); width = 0.19
    for index, (label, key) in enumerate(mapping):
        axes[0].bar(x + (index - 1.5) * width, [raw["overall"][key][m] for m in metrics], width, label=label, color=COLORS[label])
    axes[0].set_xticks(x, ["MRR", "Hits@1", "Hits@3", "Hits@10"])
    axes[0].set_ylim(0.0, 0.90); axes[0].set_ylabel("Typed filtered score")
    axes[0].set_title("(a) Confirmatory performance")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    sizes = [("1–5", "1_to_5"), ("6–10", "6_to_10"), (">10", "over_10")]
    x = np.arange(len(sizes)); width = 0.19
    for index, (label, key) in enumerate(mapping):
        axes[1].bar(x + (index - 1.5) * width, [raw["by_formula_size"][s][key + "_mrr"] for _, s in sizes], width, color=COLORS[label])
    axes[1].set_xticks(x, [label for label, _ in sizes]); axes[1].set_xlabel("Number of herbs")
    axes[1].set_ylabel("MRR"); axes[1].set_title("(b) Performance by formula size")
    save(fig, "figure_results_and_size")


def robustness():
    raw = read("expert_systems_robustness.json")
    probabilities = np.asarray(sorted(float(x) for x in raw["ingredient_dropout"]))
    means = np.asarray([raw["ingredient_dropout"][str(x)]["mrr"]["mean"] for x in probabilities])
    std = np.asarray([raw["ingredient_dropout"][str(x)]["mrr"]["std_across_masks"] for x in probabilities])
    observed = raw["composition_permutation"]["observed_mrr"]
    extended = read("composition_permutation_extended.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    axes[0].errorbar(probabilities * 100, means, yerr=std, marker="o", color="#B2182B", capsize=3)
    axes[0].axhline(observed, color="0.4", linestyle="--", label="No dropout")
    axes[0].set_xlabel("Ingredients removed (%)"); axes[0].set_ylabel("MRR")
    axes[0].set_title("(a) Missing-ingredient robustness"); axes[0].legend(frameon=False)
    axes[1].hist(extended["null_mrr_values"], bins=12, color="#92C5DE", edgecolor="white")
    axes[1].axvline(extended["observed_mrr"], color="#B2182B", linewidth=2, label="Correct composition")
    axes[1].set_xlabel("MRR"); axes[1].set_ylabel("Permutations")
    axes[1].set_title("(b) Size-matched randomization test"); axes[1].legend(frameon=False)
    save(fig, "figure_robustness")


def transfer_and_dimension():
    transfer = read("decoder_transfer_results.json")
    dims = read("dimension_sensitivity_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    decoders = ["DistMult", "TransE"]
    x = np.arange(2); width = 0.34
    mean = [transfer[d.lower()]["mean"]["summary"]["mrr"]["mean"] for d in decoders]
    pair = [transfer[d.lower()]["pair"]["summary"]["mrr"]["mean"] for d in decoders]
    mean_err = [transfer[d.lower()]["mean"]["summary"]["mrr"]["sample_std"] for d in decoders]
    pair_err = [transfer[d.lower()]["pair"]["summary"]["mrr"]["sample_std"] for d in decoders]
    axes[0].bar(x - width / 2, mean, width, yerr=mean_err, capsize=3, label="Mean", color="#2166AC")
    axes[0].bar(x + width / 2, pair, width, yerr=pair_err, capsize=3, label="Mean + pair", color="#B2182B")
    axes[0].set_xticks(x, decoders); axes[0].set_ylabel("MRR"); axes[0].set_title("(a) Decoder transfer")
    axes[0].legend(frameon=False)
    dimensions = [64, 128, 256]
    values = [dims["dimensions"][str(d)]["summary"]["mrr"]["mean"] for d in dimensions]
    errors = [dims["dimensions"][str(d)]["summary"]["mrr"]["sample_std"] for d in dimensions]
    axes[1].errorbar(dimensions, values, yerr=errors, marker="o", capsize=3, color="#B2182B")
    axes[1].set_xticks(dimensions); axes[1].set_xlabel("Embedding dimension"); axes[1].set_ylabel("MRR")
    axes[1].set_title("(b) Dimension sensitivity")
    save(fig, "figure_transfer_and_dimension")


def architecture():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.5, 3.4)); ax.set_axis_off()
    boxes = [
        (0.02, 0.36, 0.15, 0.28, "Herb set\n$\\{z_i\\}_{i=1}^n$", "#D1E5F0"),
        (0.23, 0.58, 0.20, 0.25, "First-order mean\n$\\mu=\\frac{1}{n}\\sum_i z_i$", "#E5F5E0"),
        (0.23, 0.14, 0.20, 0.25, "Pair moment, $O(nd)$\n$p=\\frac{(\\sum_i z_i)^2-\\sum_i z_i^2}{n(n-1)}$", "#FEE0D2"),
        (0.50, 0.36, 0.20, 0.28, "Size-adaptive fusion\n$z_f=\\mu+\\alpha g_u(\\mu)+\\beta(n)g_p(p)$", "#F4A582"),
        (0.77, 0.36, 0.20, 0.28, "KGE decoder\nComplEx / DistMult / TransE", "#D8DAEB"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor=color, edgecolor="0.25"))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)
    arrows = [((0.17, .50), (.23, .70)), ((.17, .50), (.23, .26)), ((.43, .70), (.50, .54)), ((.43, .26), (.50, .46)), ((.70, .50), (.77, .50))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, color="0.25"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "figure_model_architecture")


def external_validation():
    raw = read("external_zero_shot_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    methods = [
        ("Hybrid\nPairComp", "hybrid", "Hybrid PairComp"),
        ("Parameter\ncontrol", "parameter_control", "Parameter control"),
        ("DeepSets", "deepsets", "DeepSets"),
        ("Pair only", "pair_only", "Pair only"),
    ]
    x = np.arange(6)
    values = [raw["overall"][key]["mrr"]["mean"] for _, key, _ in methods]
    errors = [raw["overall"][key]["mrr"]["sample_std"] for _, key, _ in methods]
    values += [raw["frequency"]["mrr"], raw["uniform_random_expectation"]["metrics"]["mrr"]]
    errors += [0.0, 0.0]
    labels = [label for label, _, _ in methods] + ["Frequency", "Uniform\nrandom"]
    colors = [COLORS[color] for _, _, color in methods] + ["#666666", "#BDBDBD"]
    axes[0].bar(x, values, yerr=errors, capsize=3, color=colors)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("MRR"); axes[0].set_title("(a) Frozen zero-shot ranking")
    coverage = raw["coverage"]
    coverage_values = [
        coverage["herb_unique_exact_coverage"],
        coverage["herb_record_exact_coverage"],
        coverage["syndrome_unique_exact_coverage"],
        coverage["included_prescriptions"] / coverage["total_prescriptions"],
    ]
    coverage_labels = ["Herb\ntypes", "Herb\nrecords", "Syndrome\nstrings", "Included\nprescriptions"]
    axes[1].bar(np.arange(4), np.asarray(coverage_values) * 100, color=["#92C5DE", "#4393C3", "#F4A582", "#B2182B"])
    axes[1].axhline(50, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(4), coverage_labels)
    axes[1].set_ylim(0, 100); axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("(b) Exact-mapping coverage")
    save(fig, "figure_external_validation")


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    architecture(); overall_and_size(); robustness(); transfer_and_dimension(); external_validation()
    print(FIG)


if __name__ == "__main__":
    main()
