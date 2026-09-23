# -*- coding: utf-8 -*-
"""Robustness, clustered inference, and composition controls for Expert Systems."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from analyze_composition_results import typed_ranks
from analyze_hybrid_confirmatory import FAMILIES, load
from compositional_kge import OUT, make_strict_inductive, read_compositions
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)


def metrics(ranks):
    return {"mrr": float((1.0/ranks).mean()), "hits1": float((ranks <= 1).mean()),
            "hits3": float((ranks <= 3).mean()), "hits10": float((ranks <= 10).mean())}


def clustered_bootstrap(a, b, heads, seed=20261008, repetitions=20000):
    """Resample formula heads, retaining all their query edges; arrays are seed x edge."""
    delta = (1.0/a).mean(0)-(1.0/b).mean(0)
    unique = np.unique(heads)
    groups = [np.flatnonzero(heads == h) for h in unique]
    values = np.asarray([delta[g].mean() for g in groups])
    # Weight clusters by edge count to recover the edge-level estimand.
    weights = np.asarray([len(g) for g in groups], dtype=float)
    rng = np.random.default_rng(seed); samples = np.empty(repetitions)
    for i in range(repetitions):
        idx = rng.integers(0, len(groups), size=len(groups))
        samples[i] = np.average(values[idx], weights=weights[idx])
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {"formula_clusters": len(groups), "edges": len(heads),
            "mrr_difference": float(np.average(values, weights=weights)),
            "cluster_bootstrap_95ci": [float(lo), float(hi)],
            "probability_difference_le_zero": float((samples <= 0).mean()),
            "repetitions": repetitions}


def perturbed_arrays(arrays, heads, rng, dropout=None, permute=False):
    ingredients, mask, uniform, dosage, complete = arrays
    ing, msk, uni = ingredients.copy(), mask.copy(), uniform.copy()
    if dropout is not None:
        for h in np.unique(heads):
            herbs = ingredients[h, mask[h]]
            keep = rng.random(len(herbs)) >= dropout
            if not keep.any(): keep[rng.integers(len(herbs))] = True
            herbs = herbs[keep]; ing[h] = 0; msk[h] = False; uni[h] = 0
            ing[h, :len(herbs)] = herbs; msk[h, :len(herbs)] = True; uni[h, :len(herbs)] = 1.0/len(herbs)
    if permute:
        by_size = defaultdict(list)
        for f in np.flatnonzero(mask.any(1)):
            by_size[int(mask[f].sum())].append(int(f))
        for h in np.unique(heads):
            size = int(mask[h].sum()); donors = by_size[size]
            donor = int(rng.choice(donors))
            while donor == int(h) and len(donors) > 1: donor = int(rng.choice(donors))
            ing[h] = ingredients[donor]; msk[h] = mask[donor]; uni[h] = uniform[donor]
    return ing, msk, uni, dosage, complete


def hybrid_ranks(data, arrays, seed, device):
    model = HybridPairComplEx(data, "hybrid_pair", arrays).to(device)
    state = torch.load(OUT/f"hybridconfirm_hybrid_seed{seed}.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return typed_ranks(model, data, data.test, device)


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda"); data = Dataset("hybrid_holdout"); arrays = read_compositions(data)
    make_strict_inductive(data); heads = data.test[:, 0]
    ranks = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        for seed in SEEDS:
            ranks[family].append(typed_ranks(load(family, data, arrays, seed, device), data, data.test, device))
        ranks[family] = np.stack(ranks[family])
    report = {
        "clustered_bootstrap": {
            "hybrid_vs_parameter_control": clustered_bootstrap(ranks["hybrid"], ranks["parameter_control"], heads),
            "hybrid_vs_deepsets": clustered_bootstrap(ranks["hybrid"], ranks["deepsets"], heads),
            "hybrid_vs_pair_only": clustered_bootstrap(ranks["hybrid"], ranks["pair_only"], heads)},
        "by_relation": {}, "ingredient_dropout": {}, "composition_permutation": {}}
    for rid in sorted(set(map(int, data.test[:, 1]))):
        subset = data.test[:, 1] == rid
        report["by_relation"][data.relations[rid]] = {"n": int(subset.sum()),
            **{family: metrics(values[:, subset]) for family, values in ranks.items()}}

    # Missing-ingredient robustness: five deterministic masks x three model seeds.
    for probability in (0.1, 0.2, 0.3, 0.5):
        replicate_metrics = []
        for repetition in range(5):
            altered = perturbed_arrays(arrays, heads, np.random.default_rng(20261100+repetition), dropout=probability)
            rr = [hybrid_ranks(data, altered, seed, device) for seed in SEEDS]
            replicate_metrics.append(metrics(np.stack(rr)))
        report["ingredient_dropout"][str(probability)] = {
            key: {"mean": float(np.mean([x[key] for x in replicate_metrics])),
                  "std_across_masks": float(np.std([x[key] for x in replicate_metrics], ddof=1))}
            for key in ("mrr", "hits1", "hits3", "hits10")}

    # Size-matched wrong-composition negative control.
    permutation_mrr = []
    for repetition in range(10):
        altered = perturbed_arrays(arrays, heads, np.random.default_rng(20261200+repetition), permute=True)
        rr = [hybrid_ranks(data, altered, seed, device) for seed in SEEDS]
        permutation_mrr.append(metrics(np.stack(rr))["mrr"])
    observed = metrics(ranks["hybrid"])["mrr"]
    report["composition_permutation"] = {
        "observed_mrr": observed, "permutations": 10,
        "null_mrr_mean": float(np.mean(permutation_mrr)),
        "null_mrr_std": float(np.std(permutation_mrr, ddof=1)),
        "null_mrr_values": permutation_mrr,
        "empirical_p_upper": float((1+np.sum(np.asarray(permutation_mrr) >= observed))/(len(permutation_mrr)+1)),
        "preserved": "formula size and trained model", "permuted": "ingredient set across formulas"}
    (OUT/"expert_systems_robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
