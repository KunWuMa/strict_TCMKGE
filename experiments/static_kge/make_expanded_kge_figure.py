# -*- coding: utf-8 -*-
"""Create the expanded KGE baseline figure for the manuscript."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def mean_std(rows, metric):
    values = np.asarray(rows, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    expanded = json.loads(
        (RESULTS / "expanded_kge_baselines_results.json").read_text(encoding="utf-8")
    )
    hybrid = json.loads(
        (RESULTS / "hybrid_confirmatory_detailed_analysis.json").read_text(encoding="utf-8")
    )
    ingram = json.loads(
        (RESULTS / "ingram_tcm_full200_results.json").read_text(encoding="utf-8")
    )
    seeds = ("20260915", "20260916", "20260917")
    labels = [
        "PairComp-KGE", "Mean-ComplEx", "Mean-PairRE", "NodePiece-style",
        "Mean-RotatE", "InGram", "ID-TransE", "ID-DistMult", "ID-ComplEx",
    ]
    keys = [
        None, "mean_complex", "pairre", "nodepiece_transformer",
        "rotate", None, "strict_id_transe", "strict_id_distmult", "strict_id_complex",
    ]
    metrics = {"MRR": ([], []), "Hits@1": ([], [])}
    for title, field in (("MRR", "mrr"), ("Hits@1", "hits1")):
        hybrid_values = []
        for seed_value in hybrid["overall"]["hybrid"]["per_seed_mrr"]:
            hybrid_values.append(seed_value)
        if field == "hits1":
            # Aggregate Hits@1 is stored in the detailed analysis; the SD is from
            # the frozen confirmatory result file.
            frozen = json.loads(
                (RESULTS / "hybrid_confirmatory_results.json").read_text(encoding="utf-8")
            )
            hybrid_values = [
                frozen["hybrid"][seed]["typed"]["hits1"] for seed in seeds
            ]
        values = [mean_std(hybrid_values, field)]
        for key in keys[1:5]:
            model = expanded["models"][key]
            values.append(mean_std([model[seed]["typed"][field] for seed in seeds], field))
        values.append(mean_std(
            [ingram["models"][seed]["typed"][field] for seed in seeds], field
        ))
        for key in keys[6:]:
            model = expanded["models"][key]
            values.append(mean_std([model[seed]["typed"][field] for seed in seeds], field))
        metrics[title] = ([x[0] for x in values], [x[1] for x in values])

    colors = [
        "#9B1B30", "#2F6690", "#2F6690", "#3A7D44", "#2F6690",
        "#7A5195", "#8A8A8A", "#8A8A8A", "#8A8A8A",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, (title, (means, errors)) in zip(axes, metrics.items()):
        bars = ax.bar(
            x, means, yerr=errors, capsize=3, color=colors,
            edgecolor="black", linewidth=0.6,
        )
        bars[0].set_hatch("//")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x, labels, rotation=43, ha="right")
        ax.set_ylim(0, max(means) * 1.18)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_ylabel("Typed filtered score")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Strict unseen-formula link prediction (mean ± sample SD, three seeds)",
        fontsize=13,
    )
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "figure_expanded_kge_baselines.png", dpi=400)
    fig.savefig(FIGURES / "figure_expanded_kge_baselines.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
