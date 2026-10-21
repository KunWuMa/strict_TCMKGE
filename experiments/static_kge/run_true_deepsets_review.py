# -*- coding: utf-8 -*-
"""Correct Deep Sets comparison with elementwise transform before pooling."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from compositional_kge import OUT, make_strict_inductive, read_compositions
from pair_interaction_kge import PairCompositionalComplEx, train_pair
from train_baselines import Dataset, evaluate


SEEDS = (20260915, 20260916, 20260917)
VARIANT = "deepsets_elementwise"
DEV = OUT / "true_deepsets_review_dev.json"
RESULT = OUT / "true_deepsets_review_results.json"
RANK = OUT / "true_deepsets_review_ranks.npz"


def fresh():
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    return data, arrays


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    selections = json.loads(DEV.read_text(encoding="utf-8")) if DEV.exists() else {
        "architecture": "complex elementwise tanh map, mean pool, mean residual",
        "selection": "typed filtered development MRR", "seeds": SEEDS, "runs": {}}
    for seed in SEEDS:
        checkpoint = f"review_true_deepsets_seed{seed}.pt"
        if not (OUT / checkpoint).exists():
            data, arrays = fresh()
            chosen = train_pair(data, VARIANT, arrays, device, seed, checkpoint)
            selections["runs"][str(seed)] = chosen
            DEV.write_text(json.dumps(selections, indent=2), encoding="utf-8")
            print("DEV SELECTED", seed, chosen["selected"]["epoch"], flush=True)
    data, arrays = fresh()
    results = {"seeds": SEEDS, "architecture": selections["architecture"],
               "test_scored_after_all_selection": True, "runs": {}}
    ranks = []
    for seed in SEEDS:
        model = PairCompositionalComplEx(data, VARIANT, arrays).to(device)
        state = torch.load(OUT / f"review_true_deepsets_seed{seed}.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        results["runs"][str(seed)] = evaluate(model, data, data.test, device, True)
        ranks.append(typed_ranks(model, data, data.test, device))
        print("TEST", seed, results["runs"][str(seed)], flush=True)
    ranks = np.stack(ranks)
    results["summary"] = {
        metric: {"mean": float(np.mean([results["runs"][str(s)][metric] for s in SEEDS])),
                 "sample_std": float(np.std([results["runs"][str(s)][metric]
                                             for s in SEEDS], ddof=1))}
        for metric in ("mrr", "hits1", "hits3", "hits10")}
    np.savez_compressed(RANK, ranks=ranks, heads=data.test[:, 0])
    RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
