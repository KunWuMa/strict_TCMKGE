# -*- coding: utf-8 -*-
"""Regenerate manuscript figures affected by the review experiments.

All values are loaded from frozen aggregate result files. The script requires
the second-moment, true Deep Sets, external, and timing analyses to be complete.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PAPER = Path(os.environ.get("PAIRCOMP_FIGURE_DIR", HERE / "figures" / "paper"))

NAVY = "#19324a"
TEAL = "#168a83"
CORAL = "#d85f46"
BLUE = "#477ca7"
PURPLE = "#8055a0"
GOLD = "#b88621"
GRAY = "#717e88"
GRID = "#e4e9ed"
COLORS = {
    "PairComp-KGE": CORAL,
    "Second moment": GOLD,
    "Mean-ComplEx": TEAL,
    "Matched control": BLUE,
    "Deep Sets": PURPLE,
    "Set Transformer": GRAY,
    "Pair-only": NAVY,
    "Frequency": "#89a4b1",
    "Random": "#c2c9ce",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.1,
    "axes.titlesize": 10.3, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(fig, name):
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / f"{name}.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def label(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", weight="bold", pad=12)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def three_seed_scores(direct, deep, set_transformer):
    seeds = direct["seeds"][:3]
    names = ["PairComp-KGE", "Second moment", "Mean-ComplEx",
             "Matched control"]
    keys = ["paircomp", "second_moment", "mean_complex", "matched"]
    scores = {
        name: {metric: float(np.mean([
            direct["models"][key][str(seed)][metric] for seed in seeds
        ])) for metric in ("mrr", "hits1", "hits3", "hits10")}
        for name, key in zip(names, keys)
    }
    scores["Deep Sets"] = {
        metric: deep["summary"][metric]["mean"]
        for metric in ("mrr", "hits1", "hits3", "hits10")
    }
    scores["Set Transformer"] = {
        metric: set_transformer["models"]["set_transformer"]["summary"][metric]["mean"]
        for metric in ("mrr", "hits1", "hits3", "hits10")
    }
    return scores


def main_evidence(scores, analysis, set_ci):
    order = ["PairComp-KGE", "Second moment", "Mean-ComplEx",
             "Matched control", "Deep Sets", "Set Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.65),
                             gridspec_kw={"width_ratios": [1.13, 1]},
                             constrained_layout=True)
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    for yy, name in zip(y, order):
        color = COLORS[name]
        ax.plot([scores[name]["hits1"], scores[name]["mrr"]], [yy, yy],
                color=GRID, linewidth=4, zorder=1)
        ax.scatter(scores[name]["mrr"], yy, s=64, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(scores[name]["hits1"], yy, s=57, marker="s",
                   facecolor="white", edgecolor=color, linewidth=1.7, zorder=3)
    ax.set_yticks(y, order)
    ax.set_xlim(0.45, max(z["mrr"] for z in scores.values()) + 0.04)
    ax.set_xlabel("Relation-typed filtered score")
    label(ax, "a", "Top-rank performance")
    ax.plot([], [], "o", color=NAVY, label="MRR")
    ax.plot([], [], "s", markerfacecolor="white", markeredgecolor=NAVY,
            label="Hits@1")
    ax.legend(ncol=1, loc="upper left", fontsize=8.4)

    cmp = analysis["three_seed_direct_comparison"]
    effect = {
        "Second moment": cmp["paircomp_vs_second_moment"]["mrr"],
        "Mean-ComplEx": cmp["paircomp_vs_mean_complex"]["mrr"],
        "Matched control": cmp["paircomp_vs_matched"]["mrr"],
        "Deep Sets": analysis["three_seed_deepsets"]["mrr"],
        "Set Transformer": {
            "difference": set_ci["paircomp_vs_set_transformer"]["mrr_difference"],
            "cluster_bootstrap_95ci": set_ci["paircomp_vs_set_transformer"]
                                      ["cluster_bootstrap_95ci"],
        },
    }
    order = list(effect)
    y = np.arange(len(order))[::-1]
    ax = axes[1]
    ax.axvline(0, color=NAVY, linestyle="--", linewidth=1)
    limits = []
    for yy, name in zip(y, order):
        z = effect[name]
        low, high = z["cluster_bootstrap_95ci"]
        limits.extend((low, high))
        ax.plot([low, high], [yy, yy], color=COLORS[name], linewidth=3)
        ax.scatter(z["difference"], yy, color=COLORS[name], s=66,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, [f"vs {name}" for name in order])
    ax.set_xlim(min(-0.01, min(limits)-0.006), max(limits)+0.007)
    ax.set_xlabel("PairComp-KGE minus baseline MRR (95% CI)")
    label(ax, "b", "Formula-clustered effect")
    save(fig, "figure_2_main_evidence")


def size_score(analysis, key, group):
    bins = ["singleton", "2_to_5"] if group == "1–5" else [
        "6_to_10" if group == "6–10" else "over_10"]
    available = [analysis["by_formula_size"][b] for b in bins
                 if b in analysis["by_formula_size"]]
    numerator = sum(row["queries"] * row[key]["mrr"] for row in available)
    denominator = sum(row["queries"] for row in available)
    return numerator / denominator


def mechanism(scores, analysis, detailed):
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 3.95),
                             constrained_layout=True)
    groups = ["1–5", "6–10", ">10"]
    for name, key in [("PairComp-KGE", "paircomp"),
                      ("Second moment", "second_moment"),
                      ("Mean-ComplEx", "mean_complex"),
                      ("Deep Sets", "true_deepsets")]:
        values = [size_score(analysis, key, group) for group in groups]
        axes[0].plot(range(3), values, marker="o", linewidth=2.1,
                     color=COLORS[name], label=name)
    axes[0].set_xticks(range(3), groups)
    axes[0].set_xlabel("Formula size (herbs)")
    axes[0].set_ylabel("MRR")
    axes[0].legend(loc="lower left", ncol=2, fontsize=7.2)
    label(axes[0], "a", "Performance by formula size")

    gates = detailed["learned_gates"]["hybrid"]
    ns = np.arange(2, 21)
    curves = []
    for row in gates:
        at_five = np.clip(row["pair_gate_at_5"], 1e-6, 1-1e-6)
        intercept = math.log(at_five / (1-at_five))
        curves.append(1 / (1 + np.exp(-(
            intercept + row["pair_gate_log_size"] * (np.log(ns)-math.log(5))))))
    curves = np.asarray(curves)
    axes[1].fill_between(ns, curves.min(0), curves.max(0),
                         color=CORAL, alpha=0.16)
    axes[1].plot(ns, curves.mean(0), color=CORAL, linewidth=2.4,
                 label=r"Pair gate $\beta(n_f)$")
    axes[1].axhline(np.mean([x["unary_gate"] for x in gates]),
                    color=TEAL, linestyle="--", linewidth=1.7,
                    label=r"Unary gate $\alpha$")
    axes[1].set_xticks([2, 5, 10, 15, 20])
    axes[1].set_ylim(0.1, 1.02)
    axes[1].set_xlabel("Formula size $n_f$")
    axes[1].set_ylabel("Learned gate")
    axes[1].legend(fontsize=7.9)
    label(axes[1], "b", "Size-adaptive gating")

    order = ["PairComp-KGE", "Second moment", "Pair-only",
             "Deep Sets", "Matched control"]
    values = [detailed["overall"]["pair_only"]["mrr"] if name == "Pair-only"
              else scores[name]["mrr"] for name in order]
    y = np.arange(len(order))[::-1]
    axes[2].barh(y, values, color=[COLORS[name] for name in order], height=0.63)
    axes[2].set_yticks(y, order)
    axes[2].set_xlim(min(values)-0.04, max(values)+0.025)
    axes[2].set_xlabel("MRR")
    label(axes[2], "c", "Second-order controls")
    save(fig, "figure_3_mechanism")


def transfer_efficiency(analysis, efficiency):
    old = read("efficiency_results.json")
    transfer = read("decoder_transfer_analysis.json")
    dimension = read("dimension_sensitivity_results.json")
    baseline = read("expanded_kge_baselines_detailed_analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.15, 3.9),
                             constrained_layout=True)
    comps = [("ComplEx", analysis["three_seed_direct_comparison"]
              ["paircomp_vs_mean_complex"]["mrr"])]
    for decoder in ("distmult", "transe"):
        item = transfer["decoders"][decoder]["pair_vs_mean_formula_cluster_bootstrap"]
        comps.append((decoder.title(), {"difference": item["mrr_difference"],
                                       "cluster_bootstrap_95ci": item["cluster_bootstrap_95ci"]}))
    y = np.arange(3)[::-1]
    axes[0].axvline(0, color=NAVY, linestyle="--", linewidth=1)
    for yy, (name, row) in zip(y, comps):
        low, high = row["cluster_bootstrap_95ci"]
        color = TEAL if low > 0 else GRAY
        axes[0].plot([low, high], [yy, yy], color=color, linewidth=3)
        axes[0].scatter(row["difference"], yy, color=color, s=62,
                        edgecolor="white", zorder=3)
    axes[0].set_yticks(y, [name for name, _ in comps])
    axes[0].set_xlim(-0.015, 0.055)
    axes[0].set_xlabel("Pair minus mean MRR (95% CI)")
    label(axes[0], "a", "Decoder transfer")

    dims = [64, 128, 256]
    for metric, color, marker in [("mrr", CORAL, "o"), ("hits1", PURPLE, "s")]:
        means = [dimension["dimensions"][str(d)]["summary"][metric]["mean"]
                 for d in dims]
        errors = [dimension["dimensions"][str(d)]["summary"][metric]["sample_std"]
                  for d in dims]
        axes[1].errorbar(dims, means, yerr=errors, marker=marker,
                         capsize=3, linewidth=2, color=color, label=metric.upper())
    axes[1].set_xticks(dims)
    axes[1].set_ylim(0.48, 0.69)
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8)
    label(axes[1], "b", "Dimension sensitivity")

    order = [("PairComp-KGE", old["models"]["hybrid"]),
             ("Second moment", efficiency["models"]["second_moment"]),
             ("Matched control", old["models"]["parameter_control"]),
             ("Deep Sets", efficiency["models"]["true_deepsets"])]
    y = np.arange(len(order))[::-1]
    values = [row["batch_milliseconds_mean"] for _, row in order]
    errors = [row["batch_milliseconds_std"] for _, row in order]
    axes[2].barh(y, values, xerr=errors, capsize=2,
                 color=[COLORS[name] for name, _ in order], height=0.63)
    axes[2].set_yticks(y, [name for name, _ in order])
    axes[2].set_xlim(0, max(values) * 1.25)
    axes[2].set_xlabel("Batch encoding time for 610 formulas (ms)")
    label(axes[2], "c", "Encoding cost")
    save(fig, "figure_5_transfer_efficiency")


def external_stress_test():
    old = read("external_zero_shot_results.json")
    moment = read("external_second_moment_review.json")
    deep = read("external_true_deepsets_review.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.25, 3.95),
                             constrained_layout=True)
    scores = {
        "PairComp-KGE": old["overall"]["hybrid"]["mrr"]["mean"],
        "Second moment": moment["second_moment"]["mrr"]["mean"],
        "Matched control": old["overall"]["parameter_control"]["mrr"]["mean"],
        "Deep Sets": deep["true_deepsets"]["mrr"]["mean"],
        "Pair-only": old["overall"]["pair_only"]["mrr"]["mean"],
        "Frequency": old["frequency"]["mrr"],
        "Random": old["uniform_random_expectation"]["metrics"]["mrr"],
    }
    names = list(scores)
    axes[0].bar(np.arange(len(names)), list(scores.values()),
                color=[COLORS[name] for name in names], width=0.65)
    axes[0].set_xticks(np.arange(len(names)), names, rotation=42, ha="right")
    axes[0].set_ylim(0, max(scores.values()) + 0.06)
    axes[0].set_ylabel("MRR")
    label(axes[0], "a", "Frozen external ranking")

    comparisons = [
        ("Second moment", moment["paircomp_minus_second_moment"]["mrr"]),
        ("Matched control", old["patient_cluster_bootstrap"]
         ["hybrid_vs_parameter_control"]["mrr"]),
        ("Deep Sets", deep["paircomp_minus_true_deepsets"]["mrr"]),
        ("Pair-only", old["patient_cluster_bootstrap"]
         ["hybrid_vs_pair_only"]["mrr"]),
    ]
    y = np.arange(len(comparisons))[::-1]
    axes[1].axvline(0, color=NAVY, linestyle="--", linewidth=1)
    bounds = []
    for yy, (name, item) in zip(y, comparisons):
        low, high = item["patient_cluster_bootstrap_95ci"]
        bounds.extend((low, high))
        axes[1].plot([low, high], [yy, yy], color=COLORS[name], linewidth=3)
        axes[1].scatter(item["difference"], yy, s=64, color=COLORS[name],
                        edgecolor="white", zorder=3)
    axes[1].set_yticks(y, [f"vs {name}" for name, _ in comparisons])
    axes[1].set_xlim(min(bounds)-0.012, max(bounds)+0.012)
    axes[1].set_xlabel("PairComp-KGE minus baseline MRR (95% CI)")
    label(axes[1], "b", "Patient-clustered differences")

    coverage = old["coverage"]
    labels = ["Herb\ntypes", "Herb\nrecords", "Syndrome\nstrings",
              "Included\nprescriptions"]
    values = [coverage["herb_unique_exact_coverage"],
              coverage["herb_record_exact_coverage"],
              coverage["syndrome_unique_exact_coverage"],
              coverage["included_prescriptions"] / coverage["total_prescriptions"]]
    axes[2].bar(range(4), np.asarray(values)*100,
                color=[TEAL, BLUE, GOLD, CORAL], width=0.6)
    axes[2].set_xticks(range(4), labels)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Exact-mapping coverage (%)")
    label(axes[2], "c", "Mapping bottleneck")
    save(fig, "figure_6_external_stress_test")


def main():
    direct = read("second_moment_review_results.json")
    analysis = read("second_moment_review_analysis.json")
    deep = read("true_deepsets_review_results.json")
    transformer = read("expanded_kge_baselines_results.json")
    set_ci = read("set_transformer_clustered_comparison.json")
    scores = three_seed_scores(direct, deep, transformer)
    main_evidence(scores, analysis, set_ci)
    mechanism(scores, analysis, read("hybrid_confirmatory_detailed_analysis.json"))
    transfer_efficiency(analysis, read("review_encoder_efficiency.json"))
    external_stress_test()
    print(PAPER.resolve())


if __name__ == "__main__":
    main()
